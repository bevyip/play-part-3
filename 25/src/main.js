import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { causticsVertexShader, causticsFragmentShader } from './shaders/causticsShader.js';
import { bloomVertexShader, bloomFragmentShader } from './shaders/bloomShader.js';
import { poolFloorVertexShader, poolFloorFragmentShader } from './shaders/poolFloorShader.js';
import { createSwimTracker } from './swimTracker.js';

// --- Configuration & State ---
const urlParams = new URLSearchParams(window.location.search);
const initialDepthParam = parseFloat(urlParams.get('depth'));
const initialDepth = !isNaN(initialDepthParam)
  ? Math.max(0.0, Math.min(2.00, initialDepthParam))
  : 0.0;

const state = {
  depth: initialDepth,          // Current water depth in meters (0.0m starts at the 0.50m visual lightrays)
  targetDepth: initialDepth,    // Smoothly interpolated depth target
  tileSpringDepth: initialDepth,// Underdamped buoyant body depth driving camera-to-tile distance
  tileSpringVel: 0.0,           // Vertical velocity of buoyant body relative to tiles
  prevTargetDepth: initialDepth,
  prevTargetVel: 0.0,
  turnaroundWobble: 0.35,       // Buoyancy wobble envelope (spikes when depth stops before floating up)
  currentTileDistanceScale: 1.18,
  cameraPos: new THREE.Vector2(0.0, 0.0), // World XY position on pool floor in meters
  velocity: new THREE.Vector2(0.0, 0.0),  // Swimming velocity in m/s
  time: 1.5,                    // Start at an expressive wave phase
  keys: {
    w: false,
    a: false,
    s: false,
    d: false,
    ArrowUp: false,
    ArrowLeft: false,
    ArrowDown: false,
    ArrowRight: false,
  },
  isDraggingFloor: false,
  lastPointerFloor: new THREE.Vector2(),
  isDraggingHud: false,
  hasBoopedFrog: false,
  boopFreezeUntil: 0,
  wasBoopFrozen: false,
};

// --- DOM Elements ---
const canvas = document.getElementById('pool-canvas');
const frogCanvas = document.getElementById('frog-canvas');
const boopBlurOverlayEl = document.getElementById('boop-blur-overlay');
const depthHudEl = document.getElementById('depth-hud');
const depthSvgEl = document.getElementById('depth-svg');
const surfaceWavePathEl = document.getElementById('surface-wave-path');
const swimmerStickmanEl = document.getElementById('swimmer-stickman');
const stickmanArmsEl = document.getElementById('stickman-arms');
const stickmanLegsEl = document.getElementById('stickman-legs');
const frogBoopTextEl = document.getElementById('frog-boop-text');
const frogToastEl = document.getElementById('frog-toast');
const presetButtons = document.querySelectorAll('.preset-btn');

let stickmanSwimPhase = 0.0;
let lastHudTime = 0.0;
let frogToastTimeout = null;

// --- Update Right-Side Rectangular Depth Measure & Downward Swimming Stickman ---
function updateDepthUI(depthMeters, timeSec) {
  const clampedDepth = Math.max(0.0, Math.min(2.00, depthMeters));
  const dtHud = Math.max(0.0, Math.min(0.05, timeSec - lastHudTime));
  lastHudTime = timeSec;

  // Map 0.0m -> y=28 (near top surface of rectangle) and 2.0m -> y=296 (bottom of rectangle)
  const depthNorm = clampedDepth / 2.0;
  const yPos = 28 + depthNorm * 268;

  // Advance breaststroke animation faster while actively diving deeper, calmer while floating/resting
  const isDiving = state.targetDepth > state.depth + 0.008;
  const strokeSpeed = isDiving ? 6.2 : 2.1;
  stickmanSwimPhase += dtHud * strokeSpeed;

  // Breaststroke arm sweep (facing downwards: shoulder at y=3.0, head at y=7.6)
  const armSweep = 0.5 - 0.5 * Math.cos(stickmanSwimPhase); // 0 = streamline reach down, 1 = wide outward push
  const elbowX = 2.8 + armSweep * 3.0;
  const elbowY = 6.0 - armSweep * 2.4;
  const handX = 1.8 + armSweep * 7.6;
  const handY = 10.2 - armSweep * 8.4;

  if (stickmanArmsEl) {
    stickmanArmsEl.setAttribute(
      'd',
      `M ${(-handX).toFixed(1)} ${handY.toFixed(1)} L ${(-elbowX).toFixed(1)} ${elbowY.toFixed(1)} L 0 3.0 L ${elbowX.toFixed(1)} ${elbowY.toFixed(1)} L ${handX.toFixed(1)} ${handY.toFixed(1)}`
    );
  }

  // Breaststroke leg kick (extending upward from hip at y=-4.5)
  const legKick = 0.5 + 0.5 * Math.sin(stickmanSwimPhase);
  const kneeX = 1.6 + legKick * 2.2;
  const kneeY = -8.2 + legKick * 0.8;
  const footX = 2.0 + (1.0 - legKick) * 3.2;
  const footY = -12.6 + legKick * 1.4;

  if (stickmanLegsEl) {
    stickmanLegsEl.setAttribute(
      'd',
      `M ${(-footX).toFixed(1)} ${footY.toFixed(1)} L ${(-kneeX).toFixed(1)} ${kneeY.toFixed(1)} L 0 -4.5 L ${kneeX.toFixed(1)} ${kneeY.toFixed(1)} L ${footX.toFixed(1)} ${footY.toFixed(1)}`
    );
  }

  if (swimmerStickmanEl) {
    swimmerStickmanEl.setAttribute('transform', `translate(22, ${yPos.toFixed(1)})`);
  }

  // Subtle animation of the wavy water line at the top of the rectangular measure (x=4 to x=40, y=13)
  if (surfaceWavePathEl) {
    const waveAmp = 1.4;
    let waveD = `M 4 13`;
    for (let i = 1; i <= 12; i++) {
      const wx = 4 + i * 3;
      const wy = 13 + Math.sin(i * 1.1 - timeSec * 2.8) * waveAmp;
      waveD += ` L ${wx.toFixed(1)} ${wy.toFixed(1)}`;
    }
    surfaceWavePathEl.setAttribute('d', waveD);
  }
}

// --- Three.js Renderer Setup ---
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: false,
  powerPreference: 'high-performance',
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.autoClear = false;

// Dedicated transparent renderer for the 3D frog so it sits crisp above the faint blur overlay
const frogRenderer = new THREE.WebGLRenderer({
  canvas: frogCanvas,
  alpha: true,
  antialias: true,
  powerPreference: 'high-performance',
});
frogRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
frogRenderer.setSize(window.innerWidth, window.innerHeight);
frogRenderer.setClearColor(0x000000, 0.0);

// Compute visible pool floor dimensions in meters.
// In the reference portrait screenshots, ~15 tiles (1.0m) span the width and ~32 tiles span the height.
function getViewSizeMeters() {
  const aspect = window.innerWidth / window.innerHeight;
  if (aspect < 1.0) {
    const widthM = 1.0;
    const heightM = widthM / aspect;
    return new THREE.Vector2(widthM, heightM);
  } else {
    const heightM = 1.55;
    const widthM = heightM * aspect;
    return new THREE.Vector2(widthM, heightM);
  }
}

let viewSize = getViewSizeMeters();

// --- Offscreen Floating-Point Render Targets for Caustics & Underwater Scatter Bloom ---
function createRenderTargets() {
  const dpr = Math.min(window.devicePixelRatio, 2);
  const rtWidth = Math.min(2048, Math.round(window.innerWidth * dpr * 1.25));
  const rtHeight = Math.min(2048, Math.round(window.innerHeight * dpr * 1.25));

  const causticsTarget = new THREE.WebGLRenderTarget(rtWidth, rtHeight, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: false,
    stencilBuffer: false,
  });

  const bloomWidth = Math.max(256, Math.round(rtWidth * 0.5));
  const bloomHeight = Math.max(256, Math.round(rtHeight * 0.5));
  const bloomTargetH = new THREE.WebGLRenderTarget(bloomWidth, bloomHeight, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: false,
    stencilBuffer: false,
  });
  const bloomTargetV = new THREE.WebGLRenderTarget(bloomWidth, bloomHeight, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: false,
    stencilBuffer: false,
  });

  return { causticsTarget, bloomTargetH, bloomTargetV, rtWidth, rtHeight, bloomWidth, bloomHeight };
}

let rts = createRenderTargets();

// --- 1. Caustics Projection Scene (Lagrangian Refracted Light Sheet Mesh) ---
const causticsScene = new THREE.Scene();
const causticsCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);

// High-resolution surface grid (860 x 860 = ~740k vertices, ~1.48M triangles)
const GRID_SEGMENTS = 860;
const causticsGeometry = new THREE.PlaneGeometry(1.0, 1.0, GRID_SEGMENTS, GRID_SEGMENTS);

const causticsMaterial = new THREE.ShaderMaterial({
  vertexShader: causticsVertexShader,
  fragmentShader: causticsFragmentShader,
  uniforms: {
    uTime: { value: state.time },
    uDepth: { value: state.depth },
    uCameraPos: { value: new THREE.Vector2(0, 0) },
    uMeshSpan: { value: new THREE.Vector2(1.5, 1.5) },
    uGridSnap: { value: 0.005 },
    // Calibrated so primary wave crests reach focal singularity (mu1 = 0) at D = 0.95m
    uRefractionScale: { value: 1.88 },
  },
  blending: THREE.AdditiveBlending,
  depthTest: false,
  depthWrite: false,
  transparent: true,
  side: THREE.DoubleSide, // Crucial: renders folded inverted sheets (detJ < 0) when D > 0.95m!
});

const causticsMesh = new THREE.Mesh(causticsGeometry, causticsMaterial);
causticsScene.add(causticsMesh);

function updateCausticsProjectionBounds() {
  viewSize = getViewSizeMeters();
  // Caustics RT covers 1.25x the visible viewSize (inner [0.10, 0.90] UV maps to the screen)
  // and the surface mesh spans 1.50x viewSize so rays refracting inward from outside the frame never clip.
  const rtWorldW = viewSize.x * 1.25;
  const rtWorldH = viewSize.y * 1.25;

  causticsCamera.left = state.cameraPos.x - rtWorldW * 0.5;
  causticsCamera.right = state.cameraPos.x + rtWorldW * 0.5;
  causticsCamera.top = state.cameraPos.y + rtWorldH * 0.5;
  causticsCamera.bottom = state.cameraPos.y - rtWorldH * 0.5;
  causticsCamera.updateProjectionMatrix();

  const meshSpanW = viewSize.x * 1.50;
  const meshSpanH = viewSize.y * 1.50;
  causticsMaterial.uniforms.uMeshSpan.value.set(meshSpanW, meshSpanH);

  const maxSpan = Math.max(meshSpanW, meshSpanH);
  causticsMaterial.uniforms.uGridSnap.value = maxSpan / GRID_SEGMENTS;
}
updateCausticsProjectionBounds();

// --- 2. Separable Bloom Scene ---
const fullscreenQuadGeo = new THREE.PlaneGeometry(2, 2);
const bloomScene = new THREE.Scene();
const orthoQuadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

const bloomMaterial = new THREE.ShaderMaterial({
  vertexShader: bloomVertexShader,
  fragmentShader: bloomFragmentShader,
  uniforms: {
    uTexture: { value: null },
    uDirection: { value: new THREE.Vector2(1, 0) },
  },
  depthTest: false,
  depthWrite: false,
});
const bloomMesh = new THREE.Mesh(fullscreenQuadGeo, bloomMaterial);
bloomScene.add(bloomMesh);

// --- 3. Pool Floor Composite Scene ---
const floorScene = new THREE.Scene();
const floorMaterial = new THREE.ShaderMaterial({
  vertexShader: poolFloorVertexShader,
  fragmentShader: poolFloorFragmentShader,
  uniforms: {
    uCausticsTex: { value: rts.causticsTarget.texture },
    uBloomTex: { value: rts.bloomTargetV.texture },
    uCausticsTexel: { value: new THREE.Vector2(1.0 / rts.rtWidth, 1.0 / rts.rtHeight) },
    uCameraPos: { value: state.cameraPos },
    uViewSize: { value: viewSize },
    uDepth: { value: state.depth },
    uTileDistanceScale: { value: state.currentTileDistanceScale },
  },
  depthTest: false,
  depthWrite: false,
});
const floorMesh = new THREE.Mesh(fullscreenQuadGeo, floorMaterial);
floorScene.add(floorMesh);

// --- 4. 3D Frog at the Bottom of the Pool (/media/frog.glb) ---
const frogScene = new THREE.Scene();
const frogCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
frogCamera.position.set(0, 0, 5);
frogCamera.lookAt(0, 0, 0);

const frogMaterials = [];
let frogLoaded = false;
let frogWorldTopY = 0.032;

const gltfLoader = new GLTFLoader();
gltfLoader.load('./media/frog.glb', (gltf) => {
  const model = gltf.scene;
  model.updateMatrixWorld(true);

  const frogBox = new THREE.Box3();
  model.traverse((child) => {
    if (child.isMesh) {
      const name = (child.name || '').toLowerCase();
      const matName = ((child.material && child.material.name) || '').toLowerCase();
      // Hide the flat ground plane mesh bundled in frog.glb
      if (name.includes('plane') || name.includes('ground') || matName.includes('ground')) {
        child.visible = false;
        return;
      }
      frogBox.expandByObject(child);
      child.material = child.material.clone();
      child.material.transparent = true;
      child.material.opacity = 0.0;
      child.material.depthWrite = true;
      frogMaterials.push(child.material);
    }
  });

  const center = new THREE.Vector3();
  frogBox.getCenter(center);
  model.position.sub(center);

  // Orient frog top-down on the pool floor at (0, 0) at half size
  const pivot = new THREE.Group();
  pivot.add(model);
  pivot.rotation.set(Math.PI * 0.55, 0, 0);
  pivot.scale.setScalar(0.30);
  frogScene.add(pivot);

  pivot.updateMatrixWorld(true);
  const worldFrogBox = new THREE.Box3();
  model.traverse((child) => {
    if (child.isMesh && child.visible) {
      worldFrogBox.expandByObject(child);
    }
  });
  frogWorldTopY = worldFrogBox.max.y;
  frogLoaded = true;
});

function triggerFrogBoopToast() {
  // Freeze the scene at the bottom while the popups appear and hold, then unfreeze as they begin to fade out
  state.depth = 2.00;
  state.targetDepth = 2.00;
  state.tileSpringDepth = 2.00;
  state.boopFreezeUntil = performance.now() + 1750;
  state.wasBoopFrozen = true;

  if (boopBlurOverlayEl) {
    boopBlurOverlayEl.classList.add('visible');
  }
  if (frogBoopTextEl) {
    frogBoopTextEl.classList.remove('visible');
    void frogBoopTextEl.offsetWidth;
    frogBoopTextEl.classList.add('visible');
  }
  if (frogToastEl) {
    frogToastEl.classList.remove('visible');
    void frogToastEl.offsetWidth;
    frogToastEl.classList.add('visible');
  }
  if (frogToastTimeout) clearTimeout(frogToastTimeout);
  frogToastTimeout = setTimeout(() => {
    if (frogBoopTextEl) frogBoopTextEl.classList.remove('visible');
    if (frogToastEl) frogToastEl.classList.remove('visible');
  }, 2700);
}

// --- Interactive Controls (Swimming + Depth) ---
const swimTracker = createSwimTracker(state);

// 1. Keyboard Swimming (WASD / Arrow keys + Q/E for depth)
window.addEventListener('keydown', (e) => {
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (k in state.keys) {
    state.keys[k] = true;
  }
  if (k === 'q' || k === '-') {
    state.targetDepth = Math.max(0.0, state.targetDepth - 0.06);
    swimTracker.notifyManualDepthChange(state.time);
  } else if (k === 'e' || k === '=' || k === '+') {
    state.targetDepth = Math.min(2.00, state.targetDepth + 0.06);
    swimTracker.notifyManualDepthChange(state.time);
  }
});

window.addEventListener('keyup', (e) => {
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if (k in state.keys) {
    state.keys[k] = false;
  }
});

// --- Window Resize Handling ---
window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  frogRenderer.setSize(window.innerWidth, window.innerHeight);
  rts.causticsTarget.dispose();
  rts.bloomTargetH.dispose();
  rts.bloomTargetV.dispose();
  rts = createRenderTargets();

  floorMaterial.uniforms.uCausticsTex.value = rts.causticsTarget.texture;
  floorMaterial.uniforms.uBloomTex.value = rts.bloomTargetV.texture;
  floorMaterial.uniforms.uCausticsTexel.value.set(1.0 / rts.rtWidth, 1.0 / rts.rtHeight);

  updateCausticsProjectionBounds();
});

// --- Main Animation Loop ---
let lastFrameTime = performance.now();

function animate(now) {
  requestAnimationFrame(animate);

  const dt = Math.min(0.05, (now - lastFrameTime) * 0.001);
  lastFrameTime = now;

  let isBoopFrozen = now < state.boopFreezeUntil;
  if (!isBoopFrozen) {
    if (state.wasBoopFrozen) {
      state.wasBoopFrozen = false;
      if (boopBlurOverlayEl) {
        boopBlurOverlayEl.classList.remove('visible');
      }
      // As the popups begin to fade out, immediately resume floating back up
      swimTracker.triggerFloatUp(state.time);
    }
    state.time += dt;

    // Update MediaPipe breaststroke hand tracking & buoyancy float-up when enabled
    swimTracker.update(dt, state.time);

    // Smooth, calm depth interpolation for caustics & HUD
    state.depth += (state.targetDepth - state.depth) * Math.min(1.0, dt * 6.0);

    // --- Calm, Uniform-Speed Z-Axis Buoyancy for Camera-to-Tile Distance ---
    // Smoothly follow state.depth (cascaded filter eliminates any webcam frame-step jitter)
    state.tileSpringDepth += (state.depth - state.tileSpringDepth) * Math.min(1.0, dt * 5.0);

    // Trigger rising '+♥ boop!' & 'You booped the frog.' game indicators when reaching the very end
    if (state.depth >= 1.97) {
      if (!state.hasBoopedFrog) {
        state.hasBoopedFrog = true;
        triggerFrogBoopToast();
        isBoopFrozen = true;
      }
    } else if (state.depth < 1.94) {
      // Reset as soon as swimmer leaves the bottom so they can dive back down and boop again in one session
      state.hasBoopedFrog = false;
    }
  }

  if (isBoopFrozen) {
    // Freeze the scene at the bottom while the popups appear and hold for reading
    state.depth = 2.00;
    state.targetDepth = 2.00;
    state.tileSpringDepth = 2.00;
  }

  // Single, slow, steady Z-axis breathing bob (~4-second period, constant gentle amplitude)
  const calmBobOffset = Math.sin(state.time * 1.55) * 0.042;
  const buoyantTileDepth = state.tileSpringDepth + calmBobOffset;
  const tileDepthT = Math.max(-0.08, Math.min(1.06, buoyantTileDepth / 2.0));
  state.currentTileDistanceScale = Math.max(0.12, 1.24 + (0.165 - 1.24) * tileDepthT);

  if (!isBoopFrozen) {
    // Swimming keyboard acceleration + water drag
    const swimAccel = 1.35; // m/s^2
    let inputX = 0;
    let inputY = 0;
    if (state.keys.w || state.keys.ArrowUp) inputY += 1;
    if (state.keys.s || state.keys.ArrowDown) inputY -= 1;
    if (state.keys.d || state.keys.ArrowRight) inputX += 1;
    if (state.keys.a || state.keys.ArrowLeft) inputX -= 1;

    if (inputX !== 0 || inputY !== 0) {
      const len = Math.hypot(inputX, inputY);
      state.velocity.x += (inputX / len) * swimAccel * dt;
      state.velocity.y += (inputY / len) * swimAccel * dt;
    }

    // Apply position update & smooth water resistance damping
    if (!state.isDraggingFloor) {
      state.cameraPos.x += state.velocity.x * dt;
      state.cameraPos.y += state.velocity.y * dt;
    }
    const waterDrag = Math.exp(-3.8 * dt);
    state.velocity.multiplyScalar(waterDrag);
  }

  // Update camera & shader uniforms
  updateCausticsProjectionBounds();
  causticsMaterial.uniforms.uTime.value = state.time;
  causticsMaterial.uniforms.uDepth.value = state.depth;
  causticsMaterial.uniforms.uCameraPos.value.copy(state.cameraPos);

  floorMaterial.uniforms.uCameraPos.value.copy(state.cameraPos);
  floorMaterial.uniforms.uViewSize.value.copy(viewSize);
  floorMaterial.uniforms.uDepth.value = state.depth;
  floorMaterial.uniforms.uTileDistanceScale.value = state.currentTileDistanceScale;

  // Pass 1: Render Lagrangian refracted light sheet mesh into floating-point Caustics RT
  renderer.setRenderTarget(rts.causticsTarget);
  renderer.setClearColor(0x000000, 0.0);
  renderer.clear();
  renderer.render(causticsScene, causticsCamera);

  // Pass 2: Separable Underwater Scattering Bloom (Horizontal + Vertical)
  renderer.setRenderTarget(rts.bloomTargetH);
  renderer.clear();
  bloomMaterial.uniforms.uTexture.value = rts.causticsTarget.texture;
  bloomMaterial.uniforms.uDirection.value.set(1.3 / rts.bloomWidth, 0.0);
  renderer.render(bloomScene, orthoQuadCam);

  renderer.setRenderTarget(rts.bloomTargetV);
  renderer.clear();
  bloomMaterial.uniforms.uTexture.value = rts.bloomTargetH.texture;
  bloomMaterial.uniforms.uDirection.value.set(0.0, 1.3 / rts.bloomHeight);
  renderer.render(bloomScene, orthoQuadCam);

  // Pass 3: Composite Caustics + Bloom onto the Infinite Mosaic Pool Tile Floor
  renderer.setRenderTarget(null);
  renderer.render(floorScene, orthoQuadCam);

  // Pass 4: Render 3D Frog onto dedicated transparent frogCanvas (above the faint blur overlay)
  const halfW = (viewSize.x * state.currentTileDistanceScale) * 0.5;
  const halfH = (viewSize.y * state.currentTileDistanceScale) * 0.5;

  frogRenderer.clear();
  if (frogLoaded) {
    const depthNorm = Math.max(0.0, Math.min(1.0, state.depth / 2.0));
    // Stay invisible in shallow water, start fading in subtly after 0.90m (depthNorm > 0.45),
    // and reach full 1.0 opacity only right at the very end (1.99m - 2.00m)
    let rawOpacity = 0.0;
    if (depthNorm > 0.45 && depthNorm < 0.85) {
      rawOpacity = Math.pow((depthNorm - 0.45) / 0.40, 2.2) * 0.18;
    } else if (depthNorm >= 0.85) {
      rawOpacity = 0.18 + Math.pow((depthNorm - 0.85) / 0.145, 2.2) * 0.82;
    }
    const frogOpacity = Math.max(0.0, Math.min(1.0, rawOpacity));

    if (frogOpacity > 0.002) {
      frogCamera.left = state.cameraPos.x - halfW;
      frogCamera.right = state.cameraPos.x + halfW;
      frogCamera.top = state.cameraPos.y + halfH;
      frogCamera.bottom = state.cameraPos.y - halfH;
      frogCamera.updateProjectionMatrix();

      for (let i = 0; i < frogMaterials.length; i++) {
        frogMaterials[i].opacity = frogOpacity;
      }

      frogRenderer.render(frogScene, frogCamera);
    }
  }

  // Keep floating pixel-font game indicators anchored strictly above the top edge of the frog
  const frogScreenX = 50 + (-state.cameraPos.x / (halfW * 2)) * 100;
  const frogTopScreenY = 50 - ((frogWorldTopY - state.cameraPos.y) / (halfH * 2)) * 100;
  if (frogBoopTextEl) {
    frogBoopTextEl.style.left = `${frogScreenX}%`;
    frogBoopTextEl.style.top = `${frogTopScreenY - 6.2}%`;
  }
  if (frogToastEl) {
    frogToastEl.style.left = `${frogScreenX}%`;
    frogToastEl.style.top = `${frogTopScreenY - 1.6}%`;
  }

  // Update HUD & Caption
  updateDepthUI(state.depth, state.time);
}

requestAnimationFrame(animate);
