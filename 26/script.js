/**
 * Art Gallery by Beverly — Interactive Framed Art Tile Gallery
 * Four interactive exhibits where mouse proximity & hover let you "touch" the art:
 *   I.   Mimōsa Pudīca  — Sensitive plant botanical field (2D copperplate engraving)
 *   II.  Rubber Ducks   — 3D bathtub water tile with upright & side-lying bouncing ducks
 *   III. Cross Junction — 3D bird's-eye traffic intersection where cursor acts as pedestrian
 *   IV.  Comedian       — 3D duct-taped banana on wall with circular microscope / X-ray lens
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';
import { createGalleryRoom } from './room.js?v=8';
import { createPresence } from './presence.js?v=2';
(() => {
  const webglCanvas = document.getElementById('webgl-canvas');
  const canvas2d = document.getElementById('canvas-2d');
  const ctx2d = canvas2d.getContext('2d');
  const tileBezel = document.getElementById('art-tile-bezel');

  const plaqueTitle = document.getElementById('plaque-title');
  const plaqueSubtitle = document.getElementById('plaque-subtitle');
  const navPrevBtn = document.getElementById('nav-prev');
  const navNextBtn = document.getElementById('nav-next');
  const prevIconEl = document.getElementById('prev-icon');
  const nextIconEl = document.getElementById('next-icon');
  const prevLabelEl = document.getElementById('prev-label');
  const nextLabelEl = document.getElementById('next-label');

  // Exhibit museum label metadata & Mimosa drawn SVG icon (3D exhibits use static miniature 3D models)
  const EXHIBITS = [
    {
      id: 'mimosa',
      title: 'Mimōsa Pudīca',
      subtitle: 'Don\'t touch the painting!',
      shortLabel: 'Mimōsa Pudīca',
      iconSvg: `
        <svg viewBox="0 0 44 34" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M22 31C22 23 21 15 16 7" stroke="#2d3828" stroke-width="1.4" stroke-linecap="round"/>
          <path d="M21 22C25 16 30 11 36 8" stroke="#2d3828" stroke-width="1.2" stroke-linecap="round"/>
          <ellipse cx="12.7" cy="8.8" rx="4.2" ry="1.55" transform="rotate(-24 12.7 8.8)" fill="#587b59" stroke="#252e21" stroke-width="0.7"/>
          <ellipse cx="14.3" cy="12.1" rx="4.4" ry="1.55" transform="rotate(-24 14.3 12.1)" fill="#587b59" stroke="#252e21" stroke-width="0.7"/>
          <ellipse cx="15.8" cy="15.4" rx="4.2" ry="1.55" transform="rotate(-24 15.8 15.4)" fill="#587b59" stroke="#252e21" stroke-width="0.7"/>
          <ellipse cx="28" cy="12" rx="4.2" ry="1.5" transform="rotate(-18 28 12)" fill="#6a8d6b" stroke="#252e21" stroke-width="0.7"/>
          <ellipse cx="32" cy="9.5" rx="4.0" ry="1.5" transform="rotate(-18 32 9.5)" fill="#6a8d6b" stroke="#252e21" stroke-width="0.7"/>
          <ellipse cx="29" cy="16" rx="4.2" ry="1.5" transform="rotate(24 29 16)" fill="#587b59" stroke="#252e21" stroke-width="0.7"/>
          <ellipse cx="33" cy="13.5" rx="4.0" ry="1.5" transform="rotate(24 33 13.5)" fill="#587b59" stroke="#252e21" stroke-width="0.7"/>
          <circle cx="15" cy="24.5" r="4.1" fill="#d4726a" stroke="#6e2c2a" stroke-width="0.7" stroke-dasharray="1 1.2"/>
        </svg>`
    },
    {
      id: 'junction',
      title: 'Tourist in New York City',
      subtitle: '"Hey! I\'m walkin here!"',
      shortLabel: 'Tourist in New York City'
    },
    {
      id: 'ducks',
      title: 'Bath Time',
      subtitle: 'It\'s getting a little crowded in here...',
      shortLabel: 'Bath Time'
    },
    {
      id: 'banana',
      title: 'Banana?',
      subtitle: 'Look closely.',
      shortLabel: 'Banana?'
    }
  ];

  let currentExhibitIndex = 0;
  const desktopRoom = window.matchMedia('(min-width: 1024px) and (pointer: fine)').matches;
  let galleryRoom = null;
  let presence = null;
  let tileSize = 420;
  let dpr = 1;
  let lastTime = performance.now();

  // Pointer state inside the square art tile
  const pointer = {
    x: -9999,
    y: -9999,
    prevX: -9999,
    prevY: -9999,
    vx: 0,
    vy: 0,
    ndcX: 0,
    ndcY: 0,
    active: false,
    haloAlpha: 0
  };

  // Helpers
  const clamp = (v, min, max) => (v < min ? min : v > max ? max : v);
  const lerp = (a, b, t) => a + (b - a) * t;
  const smoothstep = (edge0, edge1, x) => {
    const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
    return t * t * (3 - 2 * t);
  };

  function mulberry32(seed) {
    return function () {
      let t = (seed += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function lerpRGB(c1, c2, t) {
    const r = Math.round(lerp(c1[0], c2[0], t));
    const g = Math.round(lerp(c1[1], c2[1], t));
    const b = Math.round(lerp(c1[2], c2[2], t));
    return `rgb(${r},${g},${b})`;
  }

  // Three.js shared renderer & loader
  const renderer = new THREE.WebGLRenderer({
    canvas: webglCanvas,
    antialias: true,
    alpha: false,
    powerPreference: 'high-performance'
  });
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.08;

  // Shared miniature 3D thumbnail renderer for the left/right switcher tiles
  const THUMB_W = 116;
  const THUMB_H = 92;
  const thumbRenderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true
  });
  thumbRenderer.setSize(THUMB_W, THUMB_H, false);
  thumbRenderer.outputColorSpace = THREE.SRGBColorSpace;
  thumbRenderer.toneMapping = THREE.ACESFilmicToneMapping;
  thumbRenderer.toneMappingExposure = 1.12;

  const prevThumbCanvas = document.createElement('canvas');
  prevThumbCanvas.width = THUMB_W;
  prevThumbCanvas.height = THUMB_H;
  const prevThumbCtx = prevThumbCanvas.getContext('2d');

  const nextThumbCanvas = document.createElement('canvas');
  nextThumbCanvas.width = THUMB_W;
  nextThumbCanvas.height = THUMB_H;
  const nextThumbCtx = nextThumbCanvas.getContext('2d');

  const thumbScenes = {};

  function registerThumbnailScene(id, modelGroup, camPos, lookAt, baseRotY = 0, baseRotX = 0) {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(34, THUMB_W / THUMB_H, 0.1, 30);
    camera.position.copy(camPos);
    camera.lookAt(lookAt);

    scene.add(new THREE.AmbientLight('#fffaf0', 1.75));
    const key = new THREE.DirectionalLight('#ffffff', 2.4);
    key.position.set(3, 5, 4);
    scene.add(key);
    const fill = new THREE.DirectionalLight('#e8eef5', 1.1);
    fill.position.set(-3, 2, -2);
    scene.add(fill);

    const holder = new THREE.Group();
    holder.add(modelGroup);
    holder.rotation.y = baseRotY;
    holder.rotation.x = baseRotX;
    scene.add(holder);

    thumbScenes[id] = { scene, camera, holder, baseRotY, baseRotX };
    renderSideThumbnails();
    updateGalleryUI();
  }

  function renderSideThumbnails() {
    const prevIdx = (currentExhibitIndex + EXHIBITS.length - 1) % EXHIBITS.length;
    const nextIdx = (currentExhibitIndex + 1) % EXHIBITS.length;
    const slots = [
      { ex: EXHIBITS[prevIdx], ctx: prevThumbCtx },
      { ex: EXHIBITS[nextIdx], ctx: nextThumbCtx }
    ];

    for (const slot of slots) {
      slot.ctx.clearRect(0, 0, THUMB_W, THUMB_H);
      if (slot.ex.id === 'mimosa') continue;
      const entry = thumbScenes[slot.ex.id];
      if (!entry) continue;

      entry.holder.rotation.y = entry.baseRotY;
      entry.holder.rotation.x = entry.baseRotX;

      thumbRenderer.render(entry.scene, entry.camera);
      slot.ctx.drawImage(thumbRenderer.domElement, 0, 0, THUMB_W, THUMB_H);
    }
  }

  const gltfLoader = new GLTFLoader();

  // Support KHR_materials_pbrSpecularGlossiness (used by media/person.glb) in Three.js r164
  gltfLoader.register((parser) => {
    const json = parser.json;
    if (json.materials) {
      for (const mat of json.materials) {
        const sg = mat.extensions && mat.extensions.KHR_materials_pbrSpecularGlossiness;
        if (sg) {
          const gloss = sg.glossinessFactor !== undefined ? sg.glossinessFactor : 0.25;
          mat.pbrMetallicRoughness = {
            baseColorFactor: sg.diffuseFactor || [1, 1, 1, 1],
            baseColorTexture: sg.diffuseTexture,
            metallicFactor: 0.0,
            roughnessFactor: clamp(1.0 - gloss, 0.3, 0.95)
          };
        }
      }
    }
    return { name: 'KHR_materials_pbrSpecularGlossiness' };
  });

  // ============================================================================
  // AMBIENT LAYERED AUDIO ENGINE (Bathtime Pond Quacks & NYC Street Honks)
  // ============================================================================
  const soundEngine = {
    ctx: null,
    buffers: {},
    loading: {},
    activeNodes: new Set(),
    lastMouseQuackTime: 0,
    lastDuckCollideQuackTime: 0,
    lastHonkTime: 0
  };

  const SOUND_FILES = {
    quack: 'media/quack.mp3',
    honk1: 'media/honk1.mp3',
    honk2: 'media/honk2.mp3'
  };

  function ensureAudioContext() {
    if (!soundEngine.ctx) {
      // iOS mutes Web Audio when the ring/silent switch is on unless the page asks for media playback
      if (navigator.audioSession) {
        try { navigator.audioSession.type = 'playback'; } catch (_) { }
      }
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        soundEngine.ctx = new AudioCtx();
      }
    }
    if (soundEngine.ctx && soundEngine.ctx.state !== 'running' && soundEngine.ctx.state !== 'closed') {
      soundEngine.ctx.resume().catch(() => { });
    }
    // Pre-decode sound buffers once AudioContext is available
    if (soundEngine.ctx) {
      Object.keys(SOUND_FILES).forEach((key) => {
        if (!soundEngine.buffers[key] && !soundEngine.loading[key]) {
          soundEngine.loading[key] = true;
          fetch(SOUND_FILES[key])
            .then((res) => res.arrayBuffer())
            .then((arr) => soundEngine.ctx.decodeAudioData(arr))
            .then((decoded) => {
              soundEngine.buffers[key] = decoded;
            })
            .catch(() => {
              soundEngine.loading[key] = false;
            });
        }
      });
    }
    return soundEngine.ctx;
  }

  // Mobile browsers only grant audio permission on touchend / pointerup / click, not touchstart or pointerdown
  function unlockAudio() {
    const ctx = ensureAudioContext();
    if (!ctx || ctx.state === 'running') return;
    // iOS Safari only unlocks Web Audio once a sound is started inside the gesture
    const silent = ctx.createBufferSource();
    silent.buffer = ctx.createBuffer(1, 1, 22050);
    silent.connect(ctx.destination);
    silent.start(0);
  }

  ['pointerdown', 'pointerup', 'mousedown', 'keydown', 'touchstart', 'touchend', 'click'].forEach((evt) => {
    window.addEventListener(evt, unlockAudio, { passive: true, capture: true });
  });

  function playLayeredSound(key, { volume = 0.2, rate = 1.0, fadeOutSec = 0 } = {}) {
    const ctx = ensureAudioContext();
    const buffer = soundEngine.buffers[key];
    if (ctx && buffer && ctx.state === 'running') {
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.playbackRate.value = rate;

      const gainNode = ctx.createGain();
      const now = ctx.currentTime;
      const safeVol = clamp(volume, 0.001, 0.6);

      // Soft attack to prevent any click, then optional smooth pond fade-away
      gainNode.gain.setValueAtTime(0.0008, now);
      gainNode.gain.linearRampToValueAtTime(safeVol, now + 0.018);

      if (fadeOutSec > 0) {
        const fadeEnd = now + 0.02 + fadeOutSec;
        gainNode.gain.exponentialRampToValueAtTime(0.0008, fadeEnd);
      } else {
        const dur = buffer.duration / rate;
        if (dur > 0.08) {
          gainNode.gain.setValueAtTime(safeVol, now + Math.max(0.02, dur - 0.06));
          gainNode.gain.exponentialRampToValueAtTime(0.0008, now + dur);
        }
      }

      source.connect(gainNode);
      gainNode.connect(ctx.destination);

      const entry = { source, gainNode };
      soundEngine.activeNodes.add(entry);
      source.onended = () => {
        soundEngine.activeNodes.delete(entry);
      };

      source.start(now);
      if (fadeOutSec > 0) {
        source.stop(now + 0.04 + fadeOutSec);
      }
      return;
    }

    // Fallback to HTMLAudioElement if Web Audio buffer is still decoding
    const url = SOUND_FILES[key];
    if (!url) return;
    const audio = new Audio(url);
    audio.volume = clamp(volume, 0.01, 0.6);
    audio.playbackRate = rate;
    audio.play().catch(() => { });
    if (fadeOutSec > 0) {
      const startVol = audio.volume;
      const steps = 8;
      const stepMs = (fadeOutSec * 1000) / steps;
      let step = 0;
      const timer = setInterval(() => {
        step++;
        audio.volume = Math.max(0, startVol * Math.pow(1 - step / steps, 1.8));
        if (step >= steps) {
          clearInterval(timer);
          audio.pause();
        }
      }, stepMs);
    }
  }

  function playPondQuack(isMouseContact, intensity = 1.0, residualExcite = 0) {
    const now = performance.now() * 0.001;
    if (isMouseContact) {
      if (now - soundEngine.lastMouseQuackTime < 0.11) return;
      soundEngine.lastMouseQuackTime = now;
      // Loudest when in direct contact with the mouse
      const vol = clamp(0.26 + intensity * 0.11 + (Math.random() - 0.5) * 0.04, 0.24, 0.38);
      const rate = 0.96 + Math.random() * 0.10;
      playLayeredSound('quack', { volume: vol, rate, fadeOutSec: 0 });

      // Soft staggered residual quack as the duck bobbles/bounces away from the mouse
      const echoDelayMs = 150 + Math.random() * 140;
      setTimeout(() => {
        if (EXHIBITS[currentExhibitIndex].id !== 'ducks') return;
        const echoVol = clamp(vol * (0.34 + Math.random() * 0.14), 0.06, 0.15);
        playLayeredSound('quack', {
          volume: echoVol,
          rate: 0.93 + Math.random() * 0.14,
          fadeOutSec: 0.42 + Math.random() * 0.22
        });
      }, echoDelayMs);
    } else {
      if (now - soundEngine.lastDuckCollideQuackTime < 0.075) return;
      soundEngine.lastDuckCollideQuackTime = now;
      // Softer residual pond quack when ducks bounce against each other or after a mouse collision, gently fading away
      const boost = residualExcite * 0.075;
      const vol = clamp(0.048 + intensity * 0.075 + boost + Math.random() * 0.025, 0.042, 0.175);
      const rate = 0.91 + Math.random() * 0.18;
      const fadeOutSec = 0.42 + Math.random() * 0.28;
      playLayeredSound('quack', { volume: vol, rate, fadeOutSec });

      // If this bounce carries residual energy from a mouse collision, layer a fainter secondary pond echo
      if (residualExcite > 0.22) {
        const tailDelayMs = 120 + Math.random() * 150;
        setTimeout(() => {
          if (EXHIBITS[currentExhibitIndex].id !== 'ducks') return;
          const tailVol = clamp(vol * (0.45 + Math.random() * 0.18), 0.028, 0.095);
          playLayeredSound('quack', {
            volume: tailVol,
            rate: 0.90 + Math.random() * 0.18,
            fadeOutSec: 0.36 + Math.random() * 0.24
          });
        }, tailDelayMs);
      }
    }
  }

  function playStreetHonk(level = 1.0) {
    const now = performance.now() * 0.001;
    if (now - soundEngine.lastHonkTime < 0.28) return;
    soundEngine.lastHonkTime = now;
    const key = Math.random() < 0.52 ? 'honk1' : 'honk2';
    // Slightly increased layered street volume for busy NYC street ambience
    const vol = clamp((0.145 + Math.random() * 0.14) * level, 0.11, 0.34);
    const rate = 0.96 + Math.random() * 0.08;
    playLayeredSound(key, { volume: vol, rate, fadeOutSec: 0 });
  }

  function stopAllExhibitSounds() {
    if (!soundEngine.ctx) return;
    const now = soundEngine.ctx.currentTime;
    soundEngine.activeNodes.forEach((entry) => {
      try {
        entry.gainNode.gain.cancelScheduledValues(now);
        entry.gainNode.gain.exponentialRampToValueAtTime(0.0001, now + 0.08);
        entry.source.stop(now + 0.09);
      } catch (_) { }
    });
    soundEngine.activeNodes.clear();
  }

  // ============================================================================
  // EXHIBIT I: MIMŌSA PUDĪCA (Chinese Gongbi Silk Painting + Interactive Mimosa)
  // ============================================================================
  const mimosaPaperImg = new Image();
  const mimosaStampImg = new Image();
  const mimosaState = {
    paperCanvas: document.createElement('canvas'),
    fgRockCanvas: document.createElement('canvas'),
    paperImg: mimosaPaperImg,
    stampImg: mimosaStampImg,
    stems: [],
    butterfly: {
      baseX: 0,
      baseY: 0,
      x: 0,
      y: 0,
      vx: 0,
      vy: 0,
      angle: 0,
      wingPhase: 0,
      excited: 0
    },
    radius: 95
  };

  mimosaPaperImg.onload = () => {
    if (tileSize > 0) {
      initMimosaTile();
    }
  };
  mimosaPaperImg.src = 'media/mimosa-paper.png';

  mimosaStampImg.onload = () => {
    if (tileSize > 0) {
      initMimosaTile();
    }
  };
  mimosaStampImg.src = 'media/mimosa-stamp.png';

  const MIMOSA_PALETTE = {
    inkDark: '#23281e',
    inkSoft: 'rgba(35, 40, 30, 0.72)',
    inkVein: 'rgba(30, 38, 25, 0.42)',
    inkHatch: 'rgba(26, 34, 22, 0.28)',
    stemFill: '#6b7e50',
    stemHighlight: 'rgba(182, 198, 144, 0.50)',
    stemShadow: '#465834',
    pulvinusFill: '#82915f',
    leafOpenBase: [56, 88, 58],
    leafOpenMid: [88, 122, 86],
    leafOpenTip: [118, 148, 110],
    leafClosedBase: [66, 94, 68],
    leafClosedMid: [116, 143, 112],
    leafClosedTip: [142, 166, 136],
    flowerMid: '#bf5456',
    flowerLight: '#db807e',
    flowerBlush: '#f0b6b2',
    antherTip: '#f7e6c8'
  };

  /**
   * Renders an ornamental blue-green & ochre Chinese scholar's rock (Taihu stone)
   * with craggy scalloped lobes, mineral washes, hollow windows, and malachite moss dots (diantai).
   */
  function drawScholarsRockLayer(ctx, S, isForeground, rand) {
    ctx.save();

    if (!isForeground) {
      // Back upright craggy rock spire along the left margin (x: 0..0.21*S, y: 0.42..1.0*S)
      ctx.beginPath();
      ctx.moveTo(-2, S * 0.42);
      ctx.bezierCurveTo(S * 0.03, S * 0.425, S * 0.072, S * 0.445, S * 0.088, S * 0.47);
      ctx.bezierCurveTo(S * 0.078, S * 0.495, S * 0.102, S * 0.52, S * 0.092, S * 0.555);
      ctx.bezierCurveTo(S * 0.076, S * 0.59, S * 0.112, S * 0.62, S * 0.098, S * 0.67);
      ctx.bezierCurveTo(S * 0.084, S * 0.72, S * 0.132, S * 0.76, S * 0.122, S * 0.81);
      ctx.bezierCurveTo(S * 0.115, S * 0.86, S * 0.18, S * 0.91, S * 0.20, S * 1.02);
      ctx.lineTo(-2, S * 1.02);
      ctx.closePath();

      const backGrad = ctx.createLinearGradient(0, S * 0.42, S * 0.20, S);
      backGrad.addColorStop(0, '#859684');
      backGrad.addColorStop(0.32, '#5c766e');
      backGrad.addColorStop(0.68, '#9b9875');
      backGrad.addColorStop(1, '#475c55');
      ctx.fillStyle = backGrad;
      ctx.fill();

      // Internal mineral shading & dry-brush texture washes
      ctx.save();
      ctx.clip();
      const shade1 = ctx.createRadialGradient(S * 0.04, S * 0.50, 2, S * 0.04, S * 0.50, S * 0.11);
      shade1.addColorStop(0, 'rgba(52, 80, 76, 0.58)');
      shade1.addColorStop(1, 'rgba(52, 80, 76, 0)');
      ctx.fillStyle = shade1;
      ctx.fillRect(0, S * 0.40, S * 0.24, S * 0.60);

      const shade2 = ctx.createRadialGradient(S * 0.05, S * 0.83, 4, S * 0.05, S * 0.83, S * 0.13);
      shade2.addColorStop(0, 'rgba(34, 54, 52, 0.65)');
      shade2.addColorStop(1, 'rgba(34, 54, 52, 0)');
      ctx.fillStyle = shade2;
      ctx.fillRect(0, S * 0.65, S * 0.24, S * 0.36);
      ctx.restore();

      // Delicate Gongbi contour line along the left rock spire
      ctx.strokeStyle = 'rgba(42, 48, 38, 0.58)';
      ctx.lineWidth = 1.05;
      ctx.beginPath();
      ctx.moveTo(0, S * 0.42);
      ctx.bezierCurveTo(S * 0.03, S * 0.425, S * 0.072, S * 0.445, S * 0.088, S * 0.47);
      ctx.bezierCurveTo(S * 0.078, S * 0.495, S * 0.102, S * 0.52, S * 0.092, S * 0.555);
      ctx.bezierCurveTo(S * 0.076, S * 0.59, S * 0.112, S * 0.62, S * 0.098, S * 0.67);
      ctx.bezierCurveTo(S * 0.084, S * 0.72, S * 0.132, S * 0.76, S * 0.122, S * 0.81);
      ctx.stroke();

      // Malachite moss dots (diantai) along the back rock ridge
      const backDots = [
        [0.084, 0.46], [0.090, 0.50], [0.088, 0.54], [0.080, 0.59],
        [0.072, 0.64], [0.065, 0.69], [0.058, 0.74], [0.032, 0.57],
        [0.026, 0.62], [0.048, 0.80], [0.074, 0.83]
      ];
      for (const [nx, ny] of backDots) {
        for (let k = 0; k < 4; k++) {
          const dx = (nx + (rand() - 0.5) * 0.022) * S;
          const dy = (ny + (rand() - 0.5) * 0.024) * S;
          ctx.fillStyle = k % 2 === 0 ? '#4e7d5b' : '#6c966e';
          ctx.strokeStyle = 'rgba(28, 38, 28, 0.65)';
          ctx.lineWidth = 0.55;
          ctx.beginPath();
          ctx.ellipse(dx, dy, 2.0, 1.35, rand() * Math.PI, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
      }
    } else {
      // Foreground multi-lobed craggy Taihu rock sweeping from lower-left (0.05, 1.0) to (0.50, 0.69)
      ctx.beginPath();
      ctx.moveTo(S * 0.05, S * 1.02);
      ctx.bezierCurveTo(S * 0.09, S * 0.90, S * 0.14, S * 0.81, S * 0.21, S * 0.785);
      // Scalloped upper-left lobe
      ctx.bezierCurveTo(S * 0.25, S * 0.75, S * 0.28, S * 0.705, S * 0.32, S * 0.70);
      ctx.bezierCurveTo(S * 0.345, S * 0.705, S * 0.36, S * 0.72, S * 0.38, S * 0.705);
      // Upper-right craggy crest
      ctx.bezierCurveTo(S * 0.405, S * 0.68, S * 0.45, S * 0.67, S * 0.485, S * 0.69);
      ctx.bezierCurveTo(S * 0.52, S * 0.71, S * 0.525, S * 0.745, S * 0.495, S * 0.775);
      // Lower-right undercut rock hollow
      ctx.bezierCurveTo(S * 0.455, S * 0.81, S * 0.42, S * 0.855, S * 0.385, S * 0.90);
      ctx.bezierCurveTo(S * 0.34, S * 0.94, S * 0.28, S * 0.975, S * 0.22, S * 1.02);
      ctx.closePath();

      const fgGrad = ctx.createLinearGradient(S * 0.12, S * 0.68, S * 0.46, S * 0.98);
      fgGrad.addColorStop(0, '#5d7a72');
      fgGrad.addColorStop(0.36, '#b7b18a');
      fgGrad.addColorStop(0.72, '#919471');
      fgGrad.addColorStop(1, '#4e665e');
      ctx.fillStyle = fgGrad;
      ctx.fill();

      ctx.save();
      ctx.clip();
      // Azurite/malachite blue-green mineral accents along the craggy ridges
      const ridge1 = ctx.createRadialGradient(S * 0.28, S * 0.73, 2, S * 0.28, S * 0.73, S * 0.12);
      ridge1.addColorStop(0, 'rgba(52, 86, 84, 0.70)');
      ridge1.addColorStop(1, 'rgba(52, 86, 84, 0)');
      ctx.fillStyle = ridge1;
      ctx.fillRect(S * 0.08, S * 0.65, S * 0.46, S * 0.35);

      const ridge2 = ctx.createRadialGradient(S * 0.38, S * 0.86, 2, S * 0.38, S * 0.86, S * 0.11);
      ridge2.addColorStop(0, 'rgba(44, 74, 72, 0.64)');
      ridge2.addColorStop(1, 'rgba(44, 74, 72, 0)');
      ctx.fillStyle = ridge2;
      ctx.fillRect(S * 0.18, S * 0.70, S * 0.35, S * 0.32);

      // Dry-brush vertical/diagonal mineral texture strokes inside the rock (cunfa)
      ctx.strokeStyle = 'rgba(58, 66, 52, 0.18)';
      ctx.lineWidth = 1.2;
      for (let i = 0; i < 24; i++) {
        const sx = (0.14 + rand() * 0.32) * S;
        const sy = (0.71 + rand() * 0.24) * S;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(sx - (6 + rand() * 10), sy + (10 + rand() * 14));
        ctx.stroke();
      }
      ctx.restore();

      // Crisp Gongbi rock contour lines
      ctx.strokeStyle = 'rgba(42, 48, 38, 0.56)';
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(S * 0.08, S * 0.98);
      ctx.bezierCurveTo(S * 0.11, S * 0.89, S * 0.15, S * 0.81, S * 0.21, S * 0.785);
      ctx.bezierCurveTo(S * 0.25, S * 0.75, S * 0.28, S * 0.705, S * 0.32, S * 0.70);
      ctx.bezierCurveTo(S * 0.345, S * 0.705, S * 0.36, S * 0.72, S * 0.38, S * 0.705);
      ctx.bezierCurveTo(S * 0.405, S * 0.68, S * 0.45, S * 0.67, S * 0.485, S * 0.69);
      ctx.bezierCurveTo(S * 0.52, S * 0.71, S * 0.525, S * 0.745, S * 0.495, S * 0.775);
      ctx.stroke();

      // Inner rock fissure lines
      ctx.strokeStyle = 'rgba(46, 54, 42, 0.42)';
      ctx.lineWidth = 0.9;
      ctx.beginPath();
      ctx.moveTo(S * 0.415, S * 0.76);
      ctx.bezierCurveTo(S * 0.375, S * 0.82, S * 0.335, S * 0.87, S * 0.29, S * 0.94);
      ctx.moveTo(S * 0.32, S * 0.70);
      ctx.bezierCurveTo(S * 0.295, S * 0.75, S * 0.255, S * 0.79, S * 0.215, S * 0.84);
      ctx.stroke();

      // Malachite moss dots (diantai) clustered along the scalloped rock crest
      const fgDots = [
        [0.16, 0.83], [0.20, 0.79], [0.24, 0.76], [0.28, 0.725],
        [0.32, 0.70], [0.355, 0.705], [0.40, 0.682], [0.44, 0.675],
        [0.475, 0.685], [0.33, 0.89], [0.25, 0.85]
      ];
      for (const [nx, ny] of fgDots) {
        for (let k = 0; k < 4; k++) {
          const dx = (nx + (rand() - 0.5) * 0.024) * S;
          const dy = (ny + (rand() - 0.5) * 0.022) * S;
          ctx.fillStyle = k % 2 === 0 ? '#4d7c5a' : '#6e9a6f';
          ctx.strokeStyle = 'rgba(28, 38, 28, 0.65)';
          ctx.lineWidth = 0.55;
          ctx.beginPath();
          ctx.ellipse(dx, dy, 2.0, 1.35, rand() * Math.PI, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
      }
    }

    ctx.restore();
  }

  function initMimosaTile() {
    const pCanvas = mimosaState.paperCanvas;
    pCanvas.width = tileSize * dpr;
    pCanvas.height = tileSize * dpr;
    const pCtx = pCanvas.getContext('2d', { alpha: false });

    const fgCanvas = mimosaState.fgRockCanvas;
    fgCanvas.width = tileSize * dpr;
    fgCanvas.height = tileSize * dpr;
    const fgCtx = fgCanvas.getContext('2d');
    fgCtx.clearRect(0, 0, fgCanvas.width, fgCanvas.height);

    pCtx.save();
    pCtx.scale(dpr, dpr);

    // 1. Paper ground using the uploaded paper texture (media/mimosa-paper.png)
    const rand = mulberry32(1793);
    if (
      mimosaState.paperImg &&
      mimosaState.paperImg.complete &&
      mimosaState.paperImg.naturalWidth > 0
    ) {
      pCtx.drawImage(mimosaState.paperImg, 0, 0, tileSize, tileSize);
    } else {
      const cx = tileSize * 0.5;
      const cy = tileSize * 0.5;
      const bgGrad = pCtx.createRadialGradient(cx, cy, tileSize * 0.08, cx, cy, tileSize * 0.78);
      bgGrad.addColorStop(0, '#e3dabf');
      bgGrad.addColorStop(0.65, '#dcd1b2');
      bgGrad.addColorStop(1, '#cfc19d');
      pCtx.fillStyle = bgGrad;
      pCtx.fillRect(0, 0, tileSize, tileSize);
    }

    // Traditional red cinnabar seal chop image in the bottom-right corner
    if (mimosaStampImg.complete && mimosaStampImg.naturalWidth > 0) {
      pCtx.save();
      pCtx.globalCompositeOperation = 'multiply';
      pCtx.globalAlpha = 0.54;
      const stampW = Math.round(tileSize * 0.07);
      const stampH = Math.round(
        stampW * (mimosaStampImg.naturalHeight / mimosaStampImg.naturalWidth)
      );
      const sealX = Math.round(tileSize * 0.928 - stampW);
      const sealY = Math.round(tileSize * 0.968 - stampH);
      pCtx.drawImage(mimosaStampImg, sealX, sealY, stampW, stampH);
      pCtx.restore();
    }

    // 2. Paint the background spire of the Chinese scholar's rock onto paperCanvas
    drawScholarsRockLayer(pCtx, tileSize, false, rand);
    pCtx.restore();

    // 3. Paint the foreground crag of the Chinese scholar's rock onto fgRockCanvas
    fgCtx.save();
    fgCtx.scale(dpr, dpr);
    drawScholarsRockLayer(fgCtx, tileSize, true, rand);
    fgCtx.restore();

    // 4. Cohesive Mimōsa Pudīca plant cluster spread from top-left across to bottom-right
    mimosaState.stems = [];
    mimosaState.radius = clamp(tileSize * 0.22, 85, 115);

    const fRand = mulberry32(1862);
    const unitScale = (tileSize / 430) * 0.70;

    const stemSpecs = [
      // Stem 1 (Top-Left to Upper-Center Hero Branch): reaches high into the top-left & upper-center
      {
        start: [0.08, 0.58],
        ctrl: [0.16, 0.30],
        end: [0.36, 0.17],
        scaleMult: 1.04,
        depth: 0,
        leaves: [
          { t: 0.30, side: -1, angleOff: -0.78, pinnae: 2 },
          { t: 0.62, side: -1, angleOff: -0.56, pinnae: 3 },
          { t: 0.78, side: 1, angleOff: 0.58, pinnae: 2 }
        ],
        flowers: [
          // Large upper hero blossom facing the butterfly
          { t: 0.98, side: 1, angleOff: 0.08, pedLen: 26, radius: 28.5, palette: 'pink' }
        ],
        buds: [
          { t: 0.44, angle: -Math.PI * 0.76, stalkLen: 42, budLen: 10.5, budWidth: 5.8, roseTip: true }
        ]
      },
      // Stem 2 (Mid-Left Branch): emerges from the upper rock spire into the mid-left space
      {
        start: [0.10, 0.73],
        ctrl: [0.20, 0.54],
        end: [0.31, 0.42],
        scaleMult: 0.96,
        depth: 0,
        leaves: [
          { t: 0.36, side: 1, angleOff: 0.62, pinnae: 3 },
          { t: 0.42, side: -1, angleOff: -0.68, pinnae: 2 },
          { t: 0.74, side: 1, angleOff: 0.56, pinnae: 2 }
        ],
        flowers: [
          { t: 0.98, side: 1, angleOff: 0.22, pedLen: 22, radius: 23.0, palette: 'deepPink' }
        ],
        buds: [
          { t: 0.88, angle: -0.18, stalkLen: 30, budLen: 9.0, budWidth: 5.2, roseTip: false }
        ]
      },
      // Stem 3 (Center-Right Branch): rises cleanly from behind the upper rock crest
      {
        start: [0.40, 0.70],
        ctrl: [0.51, 0.58],
        end: [0.62, 0.48],
        scaleMult: 1.0,
        depth: 0,
        leaves: [
          { t: 0.36, side: -1, angleOff: -0.68, pinnae: 2 },
          { t: 0.72, side: 1, angleOff: 0.54, pinnae: 2 }
        ],
        flowers: [
          { t: 0.98, side: -1, angleOff: -0.12, pedLen: 24, radius: 26.5, palette: 'lavender' }
        ],
        buds: [
          { t: 0.88, angle: -0.20, stalkLen: 40, budLen: 10.0, budWidth: 5.6, roseTip: true }
        ]
      },
      // Stem 4 (Bottom-Right Sweeping Branch): extends gracefully from the right side of the rock into bottom-right
      {
        start: [0.42, 0.84],
        ctrl: [0.57, 0.79],
        end: [0.74, 0.73],
        scaleMult: 0.98,
        depth: 0,
        leaves: [
          { t: 0.34, side: 1, angleOff: 0.62, pinnae: 2 },
          { t: 0.66, side: -1, angleOff: -0.58, pinnae: 2 },
          { t: 0.88, side: 1, angleOff: 0.48, pinnae: 2 }
        ],
        flowers: [
          { t: 0.98, side: -1, angleOff: -0.18, pedLen: 22, radius: 21.5, palette: 'pink' }
        ],
        buds: []
      },
      // Stem 5 (Foreground Basal Shoot): rooted at the bottom edge of the painting (y = 1.01)
      {
        start: [0.18, 1.01],
        ctrl: [0.29, 0.90],
        end: [0.38, 0.82],
        scaleMult: 0.90,
        depth: 1,
        leaves: [
          { t: 0.52, side: 1, angleOff: 0.58, pinnae: 3 },
          { t: 0.64, side: -1, angleOff: -0.58, pinnae: 2 }
        ],
        flowers: [],
        buds: [
          { t: 0.96, angle: -0.35, stalkLen: 24, budLen: 9.0, budWidth: 5.0, roseTip: true }
        ]
      }
    ];

    for (let i = 0; i < stemSpecs.length; i++) {
      mimosaState.stems.push(createCohesiveMimosaStem(stemSpecs[i], tileSize, unitScale, fRand));
    }

    // 5. Initialize the Swallowtail Butterfly in the upper-right open space (liubai)
    const b = mimosaState.butterfly;
    b.baseX = tileSize * 0.65;
    b.baseY = tileSize * 0.175;
    b.x = b.baseX;
    b.y = b.baseY;
    b.vx = 0;
    b.vy = 0;
    b.angle = 0;
    b.excited = 0;
  }

  function resetMimosaEffects() {
    for (let s = 0; s < mimosaState.stems.length; s++) {
      const stem = mimosaState.stems[s];
      for (let l = 0; l < stem.leaves.length; l++) {
        const leaf = stem.leaves[l];
        leaf.droopAngle = 0;
        for (let p = 0; p < leaf.pinnae.length; p++) {
          const pinna = leaf.pinnae[p];
          pinna.currentRelAngle = pinna.baseRelAngle;
          pinna.avgFold = 0;
          for (let k = 0; k < pinna.pairs.length; k++) {
            const pair = pinna.pairs[k];
            pair.fold = 0;
            pair.targetFold = 0;
            pair.excitation = 0;
            pair.shyTimer = 0;
          }
        }
      }
      for (let f = 0; f < stem.flowers.length; f++) {
        stem.flowers[f].shyness = 0;
      }
    }
    const b = mimosaState.butterfly;
    b.excited = 0;
    b.x = b.baseX;
    b.y = b.baseY;
    b.vx = 0;
    b.vy = 0;
    b.angle = 0;
  }

  function createCohesiveMimosaStem(spec, S, unitScale, rand) {
    const scale = unitScale * spec.scaleMult;
    const startX = spec.start[0] * S;
    const startY = spec.start[1] * S;
    const ctrlX = spec.ctrl[0] * S;
    const ctrlY = spec.ctrl[1] * S;
    const endX = spec.end[0] * S;
    const endY = spec.end[1] * S;
    const stemAngle = Math.atan2(endY - startY, endX - startX);

    const leaves = [];
    const flowers = [];
    const buds = [];
    const thorns = [];

    for (let i = 0; i < 3; i++) {
      const t = 0.20 + i * 0.26;
      const side = i % 2 === 0 ? 1 : -1;
      thorns.push({
        t,
        side,
        len: (7.5 + rand() * 5) * scale,
        angleOffset: side * (0.62 + rand() * 0.18)
      });
    }

    for (const lSpec of spec.leaves) {
      const node = evalQuadBezier(startX, startY, ctrlX, ctrlY, endX, endY, lSpec.t);
      const petioleAngle = node.angle + lSpec.angleOff;
      const petioleLen = (28 + rand() * 12) * scale;
      const numPinnae = lSpec.pinnae || 2;
      const totalFanAngle = numPinnae === 2 ? 0.92 : 1.52;

      const pinnae = [];
      for (let p = 0; p < numPinnae; p++) {
        const normP = numPinnae === 1 ? 0 : p / (numPinnae - 1) - 0.5;
        const baseRelAngle = normP * totalFanAngle + (rand() - 0.5) * 0.04;
        const outerFactor = numPinnae >= 3 ? 1 - Math.abs(normP) * 0.22 : 1;
        const pinnaLen = (74 + rand() * 16) * scale * outerFactor;
        const pinnaCurve = (normP * 0.38 + (rand() - 0.5) * 0.14) * 16 * scale;
        const numPairs = Math.max(9, Math.round((11 + Math.floor(rand() * 3)) * outerFactor));
        const pairSpacing = (pinnaLen * 0.88) / numPairs;
        const foldSide = baseRelAngle <= 0 ? 1 : -1;

        const pairs = [];
        for (let k = 0; k < numPairs; k++) {
          const u = 0.08 + ((k + 0.5) / numPairs) * 0.88;
          const profile =
            0.55 + 0.45 * Math.sin(Math.pow((k + 0.6) / (numPairs + 0.2), 0.85) * Math.PI);
          const leafletLen = (11.5 + profile * 9.5) * scale * (numPinnae === 2 ? 1.06 : 0.92);
          const leafletWidth = pairSpacing * (0.74 + profile * 0.08);

          pairs.push({
            u,
            len: leafletLen,
            width: leafletWidth,
            fold: 0,
            targetFold: 0,
            excitation: 0,
            shyTimer: 0,
            worldX: node.x,
            worldY: node.y
          });
        }

        pinnae.push({
          baseRelAngle,
          currentRelAngle: baseRelAngle,
          len: pinnaLen,
          curve: pinnaCurve,
          foldSide,
          pairs,
          avgFold: 0
        });
      }

      leaves.push({
        attachT: lSpec.t,
        side: lSpec.side,
        petioleAngle,
        petioleLen,
        droopAngle: 0,
        pinnae,
        stipuleLen: (10 + rand() * 5) * scale
      });
    }

    for (const fSpec of spec.flowers) {
      const node = evalQuadBezier(startX, startY, ctrlX, ctrlY, endX, endY, fSpec.t);
      const peduncleAngle = node.angle + fSpec.angleOff;
      const peduncleLen = (fSpec.pedLen || 24) * scale;
      const headRadius = (fSpec.radius || 20) * scale;

      const filaments = [];
      const filamentCount = 88;
      for (let i = 0; i < filamentCount; i++) {
        const layer = i / filamentCount;
        const angle = i * 2.39996 + (rand() - 0.5) * 0.14;
        const innerR = headRadius * (0.08 + (1 - layer) * 0.2);
        const outerR = headRadius * (0.70 + Math.sin(layer * Math.PI) * 0.36 + rand() * 0.14);
        const curveBend = (rand() - 0.5) * 0.24;
        const colorTier = layer < 0.32 ? 0 : layer < 0.68 ? 1 : 2;
        const hasAnther = rand() > 0.24;
        filaments.push({ angle, innerR, outerR, curveBend, colorTier, hasAnther });
      }

      flowers.push({
        attachT: fSpec.t,
        peduncleAngle,
        peduncleLen,
        headRadius,
        palette: fSpec.palette || 'pink',
        filaments,
        shyness: 0,
        worldX: node.x,
        worldY: node.y
      });
    }

    for (const bSpec of spec.buds) {
      buds.push({
        attachT: bSpec.t,
        angle: bSpec.angle,
        stalkLen: bSpec.stalkLen * scale,
        budLen: bSpec.budLen * scale,
        budWidth: bSpec.budWidth * scale,
        roseTip: bSpec.roseTip
      });
    }

    return {
      x: startX,
      y: startY,
      scale,
      depthLayer: spec.depth,
      stemAngle,
      startX,
      startY,
      ctrlX,
      ctrlY,
      endX,
      endY,
      thorns,
      leaves,
      flowers,
      buds
    };
  }

  /**
   * Renders the Gongbi Chinese Swallowtail Butterfly in the upper-right negative space.
   * Completely still on render unless the mouse hovers over/near the butterfly (b.excited > 0).
   */
  function drawMimosaButterfly(ctx, b, timeSec) {
    const S = tileSize / 430;
    const flap = b.excited > 0.002 ? 1.0 - b.excited * (0.12 - 0.12 * Math.sin(timeSec * 8.5)) : 1.0;

    ctx.save();
    ctx.translate(b.x, b.y);
    ctx.rotate(b.angle);
    ctx.scale(S, S);

    // 1. Far forewing (upper-left background wing)
    ctx.save();
    ctx.scale(1, flap * 0.95);
    const farWingGrad = ctx.createLinearGradient(-8, -52, 18, -4);
    farWingGrad.addColorStop(0, '#2a2621');
    farWingGrad.addColorStop(0.65, '#413b32');
    farWingGrad.addColorStop(1, '#26231e');
    ctx.fillStyle = farWingGrad;
    ctx.strokeStyle = '#1d1a16';
    ctx.lineWidth = 0.9;
    ctx.beginPath();
    ctx.moveTo(-6, -3);
    ctx.bezierCurveTo(-12, -26, -6, -48, 2, -55);
    ctx.bezierCurveTo(10, -52, 16, -34, 12, -8);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    // 2. Near forewing (sweeping upper-right wing with scalloped margin & fan veins)
    ctx.save();
    ctx.scale(1, flap);
    const nearWingGrad = ctx.createLinearGradient(-4, -6, 36, -44);
    nearWingGrad.addColorStop(0, '#302b24');
    nearWingGrad.addColorStop(0.55, '#595143');
    nearWingGrad.addColorStop(0.85, '#463f34');
    nearWingGrad.addColorStop(1, '#28241e');
    ctx.fillStyle = nearWingGrad;
    ctx.strokeStyle = '#1d1a16';
    ctx.lineWidth = 0.95;
    ctx.beginPath();
    ctx.moveTo(-4, -2);
    ctx.bezierCurveTo(4, -24, 18, -44, 32, -46);
    ctx.bezierCurveTo(35, -38, 31, -28, 34, -22);
    ctx.bezierCurveTo(29, -15, 24, -8, 12, -2);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Delicate Gongbi wing veins
    ctx.strokeStyle = 'rgba(24, 21, 17, 0.58)';
    ctx.lineWidth = 0.65;
    for (const [vx, vy] of [[28, -40], [30, -31], [28, -22], [23, -14]]) {
      ctx.beginPath();
      ctx.moveTo(0, -4);
      ctx.quadraticCurveTo(vx * 0.5, vy * 0.65, vx, vy);
      ctx.stroke();
    }
    ctx.restore();

    // 3. Swallowtail Hindwing (with tail projection and pale cream/jade spots)
    ctx.save();
    ctx.scale(1, 0.92 + (1 - flap) * 0.35);
    const hindGrad = ctx.createLinearGradient(0, -2, 38, 14);
    hindGrad.addColorStop(0, '#363028');
    hindGrad.addColorStop(0.6, '#27231e');
    hindGrad.addColorStop(1, '#1c1915');
    ctx.fillStyle = hindGrad;
    ctx.strokeStyle = '#1b1814';
    ctx.lineWidth = 0.95;
    ctx.beginPath();
    ctx.moveTo(-2, -1);
    ctx.bezierCurveTo(12, -8, 28, -10, 36, -6);
    // Upper tail projection
    ctx.lineTo(43, -7);
    ctx.bezierCurveTo(38, -3, 35, 1, 36, 5);
    // Main swallowtail projection
    ctx.bezierCurveTo(42, 6, 48, 7, 47, 9);
    ctx.bezierCurveTo(41, 10, 36, 9, 32, 12);
    ctx.bezierCurveTo(24, 16, 12, 12, 2, 4);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Pale cream & celadon-jade spots on hindwing (matching reference butterfly)
    const spots = [
      [18, -1, 2.1, 1.3, '#e8e2cc'],
      [21, 3, 2.2, 1.4, '#e8e2cc'],
      [25, -2.5, 1.6, 1.1, '#d5dfd4'],
      [27, 1.5, 1.8, 1.2, '#e8e2cc'],
      [26, 5.5, 1.8, 1.2, '#e8e2cc'],
      [22, 8.2, 1.6, 1.0, '#d5dfd4'],
      [31, 4.0, 1.5, 1.0, '#c67d63']
    ];
    for (const [sx, sy, rx, ry, col] of spots) {
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.ellipse(sx, sy, rx, ry, -0.2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    // 4. Slender thorax, abdomen, legs & twin curved antennae facing down-left
    ctx.fillStyle = '#24201b';
    ctx.strokeStyle = '#1a1713';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.ellipse(0, 1, 7.5, 2.4, 0.42, 0, Math.PI * 2);
    ctx.fill();

    // Delicate legs & antennae
    ctx.strokeStyle = 'rgba(34, 30, 24, 0.78)';
    ctx.lineWidth = 0.75;
    ctx.beginPath();
    // Left antenna curving up-left
    ctx.moveTo(-7, -1);
    ctx.bezierCurveTo(-16, -6, -24, -14, -27, -23);
    // Lower antenna curving left
    ctx.moveTo(-7, 0);
    ctx.bezierCurveTo(-18, -1, -26, -5, -32, -11);
    // Forelegs
    ctx.moveTo(-5, 2);
    ctx.lineTo(-10, 9);
    ctx.moveTo(-2, 3);
    ctx.lineTo(-5, 11);
    ctx.moveTo(2, 4);
    ctx.lineTo(1, 12);
    ctx.stroke();

    ctx.restore();
  }

  function evalQuadBezier(x0, y0, cx, cy, x1, y1, t) {
    const mt = 1 - t;
    const x = mt * mt * x0 + 2 * mt * t * cx + t * t * x1;
    const y = mt * mt * y0 + 2 * mt * t * cy + t * t * y1;
    const dx = 2 * mt * (cx - x0) + 2 * t * (x1 - cx);
    const dy = 2 * mt * (cy - y0) + 2 * t * (y1 - cy);
    return { x, y, angle: Math.atan2(dy, dx) };
  }

  function updateAndRenderMimosa(dt, timeSec) {
    const proxR = mimosaState.radius;
    const proxRSq = proxR * proxR;
    const stems = mimosaState.stems;

    // Physics update for Mimosa leaves and blossoms (only moves on hover/recovery)
    for (let sIdx = 0; sIdx < stems.length; sIdx++) {
      const stem = stems[sIdx];
      for (let lIdx = 0; lIdx < stem.leaves.length; lIdx++) {
        const leaf = stem.leaves[lIdx];
        let leafTotalFold = 0;
        let leafPairCount = 0;
        let maxPinnaBaseExcitation = 0;

        for (let pnIdx = 0; pnIdx < leaf.pinnae.length; pnIdx++) {
          const pinna = leaf.pinnae[pnIdx];
          const pairs = pinna.pairs;
          const n = pairs.length;
          let pinnaFoldSum = 0;

          if (pointer.active) {
            for (let k = 0; k < n; k++) {
              const pair = pairs[k];
              const dx = pair.worldX - pointer.x;
              const dy = pair.worldY - pointer.y;
              const dSq = dx * dx + dy * dy;
              if (dSq < proxRSq) {
                const dist = Math.sqrt(dSq);
                const strength = smoothstep(proxR, proxR * 0.2, dist);
                if (strength > 0.02) {
                  pair.excitation = Math.max(pair.excitation, strength);
                  pair.shyTimer = Math.max(pair.shyTimer, 1.0 + strength * 1.1 + (k / n) * 0.25);
                }
              }
            }
          }

          // Domino wave along rachis
          const waveTransfer = clamp(dt * 11.0, 0, 0.86);
          for (let k = 0; k < n - 1; k++) {
            if (pairs[k].excitation > 0.2 && pairs[k + 1].excitation < pairs[k].excitation * 0.96) {
              pairs[k + 1].excitation = lerp(
                pairs[k + 1].excitation,
                pairs[k].excitation * 0.96,
                waveTransfer
              );
              pairs[k + 1].shyTimer = Math.max(pairs[k + 1].shyTimer, pairs[k].shyTimer - 0.04);
            }
          }
          for (let k = n - 1; k > 0; k--) {
            if (pairs[k].excitation > 0.2 && pairs[k - 1].excitation < pairs[k].excitation * 0.96) {
              pairs[k - 1].excitation = lerp(
                pairs[k - 1].excitation,
                pairs[k].excitation * 0.96,
                waveTransfer
              );
              pairs[k - 1].shyTimer = Math.max(pairs[k - 1].shyTimer, pairs[k].shyTimer - 0.04);
            }
          }

          if (n > 0 && pairs[0].excitation > maxPinnaBaseExcitation) {
            maxPinnaBaseExcitation = pairs[0].excitation;
          }

          for (let k = 0; k < n; k++) {
            const pair = pairs[k];
            if (pair.shyTimer > 0) {
              pair.shyTimer = Math.max(0, pair.shyTimer - dt);
              pair.targetFold = clamp(pair.excitation, 0, 1);
            } else {
              pair.excitation = Math.max(0, pair.excitation - dt * 0.36);
              pair.targetFold = pair.excitation;
            }
            const rate = pair.targetFold > pair.fold ? 3.8 : 0.85;
            pair.fold = lerp(pair.fold, pair.targetFold, clamp(dt * rate, 0, 1));
            pinnaFoldSum += pair.fold;
          }

          pinna.avgFold = n > 0 ? pinnaFoldSum / n : 0;
          leafTotalFold += pinnaFoldSum;
          leafPairCount += n;
        }

        if (maxPinnaBaseExcitation > 0.5 && leaf.pinnae.length > 1) {
          const cross = maxPinnaBaseExcitation * 0.85;
          for (let pnIdx = 0; pnIdx < leaf.pinnae.length; pnIdx++) {
            const bp = leaf.pinnae[pnIdx].pairs[0];
            if (bp && bp.excitation < cross) {
              bp.excitation = lerp(bp.excitation, cross, clamp(dt * 3.6, 0, 1));
              bp.shyTimer = Math.max(bp.shyTimer, 0.9);
            }
          }
        }

        const leafAvgFold = leafPairCount > 0 ? leafTotalFold / leafPairCount : 0;
        leaf.droopAngle = lerp(
          leaf.droopAngle,
          leafAvgFold * 0.18 * (leaf.side || 1),
          clamp(dt * 2.8, 0, 1)
        );
        for (let pnIdx = 0; pnIdx < leaf.pinnae.length; pnIdx++) {
          const pinna = leaf.pinnae[pnIdx];
          const conv = 1 - (pinna.avgFold * 0.28 + leafAvgFold * 0.14);
          pinna.currentRelAngle = lerp(
            pinna.currentRelAngle,
            pinna.baseRelAngle * conv,
            clamp(dt * 3.2, 0, 1)
          );
        }
      }

      for (let fIdx = 0; fIdx < stem.flowers.length; fIdx++) {
        const flower = stem.flowers[fIdx];
        let targetShy = 0;
        if (pointer.active) {
          const dSq =
            (flower.worldX - pointer.x) ** 2 + (flower.worldY - pointer.y) ** 2;
          if (dSq < proxRSq) {
            targetShy = smoothstep(proxR, 12, Math.sqrt(dSq));
          }
        }
        flower.shyness = lerp(
          flower.shyness,
          targetShy,
          clamp(dt * (targetShy > flower.shyness ? 4.2 : 1.1), 0, 1)
        );
      }
    }

    // Update Swallowtail Butterfly: completely static unless pointer hovers over/near it
    const b = mimosaState.butterfly;
    let bTargetX = b.baseX;
    let bTargetY = b.baseY;
    let bTargetExcited = 0;
    if (pointer.active) {
      const dx = b.baseX - pointer.x;
      const dy = b.baseY - pointer.y;
      const dist = Math.hypot(dx, dy);
      if (dist < proxR * 0.85 && dist > 0.1) {
        bTargetExcited = smoothstep(proxR * 0.85, 14, dist);
        const push = bTargetExcited * 14;
        bTargetX += (dx / dist) * push + Math.sin(timeSec * 2.2) * 3.2 * bTargetExcited;
        bTargetY += (dy / dist) * push + Math.cos(timeSec * 2.6) * 2.6 * bTargetExcited;
      }
    }
    b.excited = lerp(b.excited, bTargetExcited, clamp(dt * 4.5, 0, 1));
    if (b.excited < 0.002) {
      b.excited = 0;
      b.x = b.baseX;
      b.y = b.baseY;
      b.angle = 0;
    } else {
      b.x = lerp(b.x, bTargetX, clamp(dt * 4.0, 0, 1));
      b.y = lerp(b.y, bTargetY, clamp(dt * 4.0, 0, 1));
      b.angle = Math.sin(timeSec * 2.4) * 0.045 * b.excited;
    }

    // Render onto canvas2d
    ctx2d.save();
    ctx2d.scale(dpr, dpr);
    ctx2d.drawImage(mimosaState.paperCanvas, 0, 0, tileSize, tileSize);

    // 1. Draw stems emerging from behind the foreground rock crag (depthLayer === 0)
    for (let i = 0; i < stems.length; i++) {
      if (stems[i].depthLayer === 0) {
        drawMimosaBranch(ctx2d, stems[i], timeSec);
      }
    }

    // 2. Draw the foreground scholar's rock crag overlapping the base of the main stems
    ctx2d.drawImage(mimosaState.fgRockCanvas, 0, 0, tileSize, tileSize);

    // 3. Draw foreground basal stems & leaves in front of the rock (depthLayer === 1)
    for (let i = 0; i < stems.length; i++) {
      if (stems[i].depthLayer > 0) {
        drawMimosaBranch(ctx2d, stems[i], timeSec);
      }
    }

    // 4. Draw the Chinese swallowtail butterfly in the upper-right negative space
    drawMimosaButterfly(ctx2d, b, timeSec);

    ctx2d.restore();
  }

  function drawMimosaBranch(ctx, stem, timeSec) {
    const scale = stem.scale;
    ctx.save();

    // Thorns
    for (let i = 0; i < stem.thorns.length; i++) {
      const th = stem.thorns[i];
      const pt = evalQuadBezier(
        stem.startX,
        stem.startY,
        stem.ctrlX,
        stem.ctrlY,
        stem.endX,
        stem.endY,
        th.t
      );
      const ang = pt.angle + th.angleOffset;
      const tx = pt.x + Math.cos(ang) * th.len;
      const ty = pt.y + Math.sin(ang) * th.len;
      const px = -Math.sin(ang) * (1.2 * scale);
      const py = Math.cos(ang) * (1.2 * scale);
      ctx.fillStyle = MIMOSA_PALETTE.pulvinusFill;
      ctx.strokeStyle = MIMOSA_PALETTE.inkDark;
      ctx.lineWidth = 0.7 * scale;
      ctx.beginPath();
      ctx.moveTo(pt.x + px, pt.y + py);
      ctx.lineTo(tx, ty);
      ctx.lineTo(pt.x - px, pt.y - py);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }

    // Stem
    ctx.lineCap = 'round';
    ctx.strokeStyle = MIMOSA_PALETTE.inkDark;
    ctx.lineWidth = 3.6 * scale;
    ctx.beginPath();
    ctx.moveTo(stem.startX, stem.startY);
    ctx.quadraticCurveTo(stem.ctrlX, stem.ctrlY, stem.endX, stem.endY);
    ctx.stroke();

    ctx.strokeStyle = MIMOSA_PALETTE.stemFill;
    ctx.lineWidth = 2.2 * scale;
    ctx.beginPath();
    ctx.moveTo(stem.startX, stem.startY);
    ctx.quadraticCurveTo(stem.ctrlX, stem.ctrlY, stem.endX, stem.endY);
    ctx.stroke();

    // Buds (static on render)
    for (let b = 0; b < stem.buds.length; b++) {
      const bud = stem.buds[b];
      const node = evalQuadBezier(
        stem.startX,
        stem.startY,
        stem.ctrlX,
        stem.ctrlY,
        stem.endX,
        stem.endY,
        bud.attachT
      );
      const ang = bud.angle;
      const bx = node.x + Math.cos(ang) * bud.stalkLen;
      const by = node.y + Math.sin(ang) * bud.stalkLen;
      ctx.strokeStyle = MIMOSA_PALETTE.inkDark;
      ctx.lineWidth = 1.2 * scale;
      ctx.beginPath();
      ctx.moveTo(node.x, node.y);
      ctx.lineTo(bx, by);
      ctx.stroke();

      ctx.save();
      ctx.translate(bx, by);
      ctx.rotate(ang);
      ctx.fillStyle = bud.roseTip ? '#b66872' : '#6c8a64';
      ctx.strokeStyle = MIMOSA_PALETTE.inkDark;
      ctx.lineWidth = 0.7 * scale;
      ctx.beginPath();
      ctx.ellipse(
        bud.budLen * 0.5,
        0,
        bud.budLen * 0.55,
        bud.budWidth * 0.55,
        0,
        0,
        Math.PI * 2
      );
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    // Compound Leaves (static unless folding/unfolding on hover)
    for (let l = 0; l < stem.leaves.length; l++) {
      const leaf = stem.leaves[l];
      const node = evalQuadBezier(
        stem.startX,
        stem.startY,
        stem.ctrlX,
        stem.ctrlY,
        stem.endX,
        stem.endY,
        leaf.attachT
      );
      const petAngle = leaf.petioleAngle + leaf.droopAngle;
      const petEndX = node.x + Math.cos(petAngle) * leaf.petioleLen;
      const petEndY = node.y + Math.sin(petAngle) * leaf.petioleLen;

      ctx.strokeStyle = MIMOSA_PALETTE.inkDark;
      ctx.lineWidth = 2.4 * scale;
      ctx.beginPath();
      ctx.moveTo(node.x, node.y);
      ctx.lineTo(petEndX, petEndY);
      ctx.stroke();

      ctx.strokeStyle = MIMOSA_PALETTE.stemFill;
      ctx.lineWidth = 1.35 * scale;
      ctx.beginPath();
      ctx.moveTo(node.x, node.y);
      ctx.lineTo(petEndX, petEndY);
      ctx.stroke();

      drawMimosaPulvinus(ctx, node.x, node.y, petAngle, 6.2 * scale, 3.3 * scale, scale);

      for (let p = 0; p < leaf.pinnae.length; p++) {
        drawMimosaPinna(ctx, petEndX, petEndY, petAngle, leaf.pinnae[p], scale);
      }
    }

    // Globose Flowers (do not shrink in size; sway gently only when hovered on)
    for (let f = 0; f < stem.flowers.length; f++) {
      const flower = stem.flowers[f];
      const node = evalQuadBezier(
        stem.startX,
        stem.startY,
        stem.ctrlX,
        stem.ctrlY,
        stem.endX,
        stem.endY,
        flower.attachT
      );
      const sway =
        flower.shyness > 0.002
          ? Math.sin(timeSec * 4.5 + flower.attachT * 10) * 0.095 * flower.shyness
          : 0;
      const pedAngle = flower.peduncleAngle + sway;
      const headX = node.x + Math.cos(pedAngle) * flower.peduncleLen;
      const headY = node.y + Math.sin(pedAngle) * flower.peduncleLen;
      flower.worldX = headX;
      flower.worldY = headY;

      ctx.strokeStyle = MIMOSA_PALETTE.inkDark;
      ctx.lineWidth = 1.8 * scale;
      ctx.beginPath();
      ctx.moveTo(node.x, node.y);
      ctx.lineTo(headX, headY);
      ctx.stroke();

      ctx.strokeStyle = MIMOSA_PALETTE.stemFill;
      ctx.lineWidth = 0.95 * scale;
      ctx.beginPath();
      ctx.moveTo(node.x, node.y);
      ctx.lineTo(headX, headY);
      ctx.stroke();

      ctx.save();
      ctx.translate(headX, headY);
      ctx.rotate(pedAngle);

      const isLav = flower.palette === 'lavender';
      const isDeep = flower.palette === 'deepPink';
      const r = flower.headRadius;
      const washGrad = ctx.createRadialGradient(0, 0, r * 0.08, 0, 0, r * 1.05);
      if (isLav) {
        washGrad.addColorStop(0, 'rgba(116, 98, 134, 0.82)');
        washGrad.addColorStop(0.55, 'rgba(168, 152, 186, 0.48)');
        washGrad.addColorStop(1, 'rgba(216, 208, 226, 0)');
      } else {
        washGrad.addColorStop(0, isDeep ? 'rgba(148, 52, 62, 0.86)' : 'rgba(156, 68, 74, 0.82)');
        washGrad.addColorStop(0.55, 'rgba(210, 114, 118, 0.48)');
        washGrad.addColorStop(1, 'rgba(238, 184, 184, 0)');
      }
      ctx.fillStyle = washGrad;
      ctx.beginPath();
      ctx.ellipse(0, 0, r * 1.1, r * 0.92, 0, 0, Math.PI * 2);
      ctx.fill();

      const cMid = isLav ? '#8f7ca8' : isDeep ? '#b64854' : MIMOSA_PALETTE.flowerMid;
      const cLight = isLav ? '#b4a5ca' : isDeep ? '#d4727c' : MIMOSA_PALETTE.flowerLight;
      const cBlush = isLav ? '#dcd4ea' : MIMOSA_PALETTE.flowerBlush;

      for (let i = 0; i < flower.filaments.length; i++) {
        const fil = flower.filaments[i];
        const cosA = Math.cos(fil.angle) * 1.08;
        const sinA = Math.sin(fil.angle) * 0.88;
        const r0 = fil.innerR;
        const r1 = fil.outerR;
        const midR = (r0 + r1) * 0.5;
        const bend = fil.angle + fil.curveBend + sway * 0.65;

        ctx.strokeStyle =
          fil.colorTier === 0 ? cMid : fil.colorTier === 1 ? cLight : cBlush;
        ctx.lineWidth = (fil.colorTier === 0 ? 0.75 : 0.62) * scale;
        ctx.beginPath();
        ctx.moveTo(cosA * r0, sinA * r0);
        ctx.quadraticCurveTo(
          Math.cos(bend) * 1.08 * midR,
          Math.sin(bend) * 0.88 * midR,
          cosA * r1,
          sinA * r1
        );
        ctx.stroke();

        if (fil.hasAnther) {
          ctx.fillStyle = MIMOSA_PALETTE.antherTip;
          ctx.beginPath();
          ctx.arc(cosA * r1, sinA * r1, 0.65 * scale, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      ctx.restore();
    }

    ctx.restore();
  }

  function drawMimosaPulvinus(ctx, x, y, angle, length, thickness, scale) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.fillStyle = MIMOSA_PALETTE.pulvinusFill;
    ctx.strokeStyle = MIMOSA_PALETTE.inkDark;
    ctx.lineWidth = 0.72 * scale;
    ctx.beginPath();
    ctx.ellipse(length * 0.5, 0, length * 0.55, thickness * 0.55, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  function drawMimosaPinna(ctx, originX, originY, petioleAngle, pinna, scale) {
    const pinnaAngle = petioleAngle + pinna.currentRelAngle;
    const foldCurveBoost = 1 + pinna.avgFold * 0.25 * pinna.foldSide;
    const endX = originX + Math.cos(pinnaAngle) * pinna.len;
    const endY = originY + Math.sin(pinnaAngle) * pinna.len;
    const ctrlX = (originX + endX) * 0.5 - Math.sin(pinnaAngle) * pinna.curve * foldCurveBoost;
    const ctrlY = (originY + endY) * 0.5 + Math.cos(pinnaAngle) * pinna.curve * foldCurveBoost;

    drawMimosaPulvinus(ctx, originX, originY, pinnaAngle, 5.0 * scale, 2.6 * scale, scale);

    ctx.strokeStyle = MIMOSA_PALETTE.inkDark;
    ctx.lineWidth = 1.8 * scale;
    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.quadraticCurveTo(ctrlX, ctrlY, endX, endY);
    ctx.stroke();

    ctx.strokeStyle = MIMOSA_PALETTE.stemFill;
    ctx.lineWidth = 0.95 * scale;
    ctx.beginPath();
    ctx.moveTo(originX, originY);
    ctx.quadraticCurveTo(ctrlX, ctrlY, endX, endY);
    ctx.stroke();

    const pairs = pinna.pairs;
    const foldSide = pinna.foldSide;
    const openAngle = 1.24;
    const closedDorsalAngle = 0.44;
    const closedVentralAngle = 0.36;

    for (let k = 0; k < pairs.length; k++) {
      const pair = pairs[k];
      const pt = evalQuadBezier(originX, originY, ctrlX, ctrlY, endX, endY, pair.u);
      pair.worldX = pt.x;
      pair.worldY = pt.y;

      const f = pair.fold;
      const fe = f * f * (3 - 2 * f);
      const rachisAngle = pt.angle;

      const angleDorsal = rachisAngle + foldSide * lerp(openAngle, closedDorsalAngle, fe);
      const ventralRoll = Math.pow(fe, 0.72);
      const ventralRelAngle = lerp(
        -foldSide * openAngle,
        foldSide * closedVentralAngle,
        ventralRoll
      );
      const angleVentral = rachisAngle + ventralRelAngle;

      const crossProgress = clamp(Math.abs(ventralRelAngle) / openAngle, 0.48, 1);
      const ventralWidth =
        pair.width * (f < 0.78 ? lerp(1, crossProgress, Math.sin(fe * Math.PI)) : 1.36);
      const dorsalWidth = pair.width * (1 + fe * 0.52);

      drawMimosaLeaflet(
        ctx,
        pt.x,
        pt.y,
        angleVentral,
        pair.len,
        ventralWidth,
        -foldSide,
        fe,
        scale,
        false
      );
      drawMimosaLeaflet(
        ctx,
        pt.x,
        pt.y,
        angleDorsal,
        pair.len,
        dorsalWidth,
        foldSide,
        fe,
        scale,
        true
      );
    }
  }

  function drawMimosaLeaflet(ctx, x, y, angle, len, width, side, fold, scale, isDorsal) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);

    const hw = width * 0.5;
    const l = len;
    const colorShift = isDorsal ? fold * 0.95 : fold * 0.8;
    const baseColor = lerpRGB(
      MIMOSA_PALETTE.leafOpenBase,
      MIMOSA_PALETTE.leafClosedBase,
      colorShift
    );
    const midColor = lerpRGB(MIMOSA_PALETTE.leafOpenMid, MIMOSA_PALETTE.leafClosedMid, colorShift);
    const tipColor = lerpRGB(MIMOSA_PALETTE.leafOpenTip, MIMOSA_PALETTE.leafClosedTip, colorShift);

    const grad = ctx.createLinearGradient(0, 0, l, 0);
    grad.addColorStop(0, baseColor);
    grad.addColorStop(0.5, midColor);
    grad.addColorStop(1, tipColor);

    const stalk = 0.85 * scale;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(stalk, -hw * 0.45);
    ctx.bezierCurveTo(l * 0.08, -hw, l * 0.76, -hw * 0.98, l * 0.95, -hw * 0.28);
    ctx.quadraticCurveTo(l * 1.02, 0, l * 0.95, hw * 0.28);
    ctx.bezierCurveTo(l * 0.76, hw * 0.98, l * 0.08, hw, stalk, hw * 0.45);
    ctx.closePath();

    ctx.fillStyle = grad;
    ctx.fill();
    ctx.strokeStyle = MIMOSA_PALETTE.inkDark;
    ctx.lineWidth = (0.62 + fold * 0.14) * scale;
    ctx.stroke();

    const midribOffset = fold * hw * 0.24 * -side;
    ctx.strokeStyle = MIMOSA_PALETTE.inkVein;
    ctx.lineWidth = 0.45 * scale;
    ctx.beginPath();
    ctx.moveTo(stalk, 0);
    ctx.quadraticCurveTo(l * 0.5, midribOffset, l * 0.93, 0);
    ctx.stroke();

    ctx.restore();
  }

  // ============================================================================
  // EXHIBIT II: RUBBER DUCKS ON WATER TILE (Three.js + media/rubber_duck.glb)
  // ============================================================================
  const rippleSettings = {
    maxSize: 54,
    animationSpeed: 2.6,
    strokeColor: [188, 234, 255]
  };

  const duckRippleCanvas = document.createElement('canvas');
  duckRippleCanvas.width = 512;
  duckRippleCanvas.height = 512;
  const duckRippleTex = new THREE.CanvasTexture(duckRippleCanvas);

  const duckState = {
    scene: new THREE.Scene(),
    camera: new THREE.PerspectiveCamera(34, 1, 0.1, 100),
    ducks: [],
    ripples: [],
    rippleCanvas: duckRippleCanvas,
    rippleTex: duckRippleTex,
    lastRippleX: -9999,
    lastRippleY: -9999,
    waterPlane: new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
    raycaster: new THREE.Raycaster(),
    pointerWorld: new THREE.Vector3(999, 0, 999),
    loaded: false,
    causticUniforms: {
      uTime: { value: 0 }
    }
  };

  function createCeramicTileFloorTexture() {
    const c = document.createElement('canvas');
    c.width = 512;
    c.height = 512;
    const g = c.getContext('2d');

    // Deep turquoise ceramic pool base
    g.fillStyle = '#2d9da8';
    g.fillRect(0, 0, 512, 512);

    // Subtle porcelain mosaic tile grid & bevels
    const step = 32;
    for (let y = 0; y < 512; y += step) {
      for (let x = 0; x < 512; x += step) {
        g.fillStyle = (x / step + y / step) % 2 === 0 ? '#32a6b1' : '#2c9aa5';
        g.fillRect(x + 1, y + 1, step - 2, step - 2);
      }
    }
    g.strokeStyle = 'rgba(215, 250, 255, 0.14)';
    g.lineWidth = 1.5;
    for (let i = 0; i <= 512; i += step) {
      g.beginPath();
      g.moveTo(i, 0);
      g.lineTo(i, 512);
      g.stroke();
      g.beginPath();
      g.moveTo(0, i);
      g.lineTo(512, i);
      g.stroke();
    }

    const tex = new THREE.CanvasTexture(c);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(2, 2);
    tex.colorSpace = THREE.SRGBColorSpace;
    return tex;
  }

  function setupDuckScene() {
    const { scene, camera } = duckState;
    scene.background = new THREE.Color('#34a6b1');

    // Camera perched at a slight gallery angle above the bathtub water tile
    camera.position.set(0, 11.2, 4.4);
    camera.lookAt(0, 0, 0.15);

    const ambLight = new THREE.AmbientLight('#fff9eb', 1.35);
    scene.add(ambLight);

    const dirLight = new THREE.DirectionalLight('#fffdf6', 2.2);
    dirLight.position.set(6, 14, 7);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.set(1024, 1024);
    dirLight.shadow.camera.near = 2;
    dirLight.shadow.camera.far = 30;
    const d = 6.5;
    dirLight.shadow.camera.left = -d;
    dirLight.shadow.camera.right = d;
    dirLight.shadow.camera.top = d;
    dirLight.shadow.camera.bottom = -d;
    dirLight.shadow.bias = -0.001;
    scene.add(dirLight);

    const rimLight = new THREE.DirectionalLight('#8be8f0', 0.9);
    rimLight.position.set(-7, 8, -6);
    scene.add(rimLight);

    // 1. Porcelain mosaic floor that receives 3D shadows from all floating ducks
    const poolFloorGeo = new THREE.PlaneGeometry(14, 14);
    const poolFloorMat = new THREE.MeshStandardMaterial({
      map: createCeramicTileFloorTexture(),
      roughness: 0.42,
      metalness: 0.05
    });
    const poolFloor = new THREE.Mesh(poolFloorGeo, poolFloorMat);
    poolFloor.rotation.x = -Math.PI / 2;
    poolFloor.position.y = -0.56;
    poolFloor.receiveShadow = true;
    scene.add(poolFloor);

    // 2. Soft, Subdued Underwater Caustics & Gentle Diffused Sunbeams Overlay
    const causticMat = new THREE.ShaderMaterial({
      uniforms: duckState.causticUniforms,
      transparent: true,
      depthWrite: false,
      blending: THREE.NormalBlending,
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float uTime;
        varying vec2 vUv;

        vec2 hash2(vec2 p) {
          return fract(sin(vec2(
            dot(p, vec2(127.1, 311.7)),
            dot(p, vec2(269.5, 183.3))
          )) * 43758.5453);
        }

        // Soft, rounded Voronoi caustic cells without sharp electrical spikes
        float softVoronoiCaustic(vec2 x, float time) {
          vec2 n = floor(x);
          vec2 f = fract(x);
          float f1 = 8.0;
          float f2 = 8.0;
          for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
              vec2 g = vec2(float(i), float(j));
              vec2 o = hash2(n + g);
              o = 0.5 + 0.38 * sin(time + 6.2831 * o);
              vec2 r = g - f + o;
              float d = dot(r, r);
              if (d < f1) {
                f2 = f1;
                f1 = d;
              } else if (d < f2) {
                f2 = d;
              }
            }
          }
          float edge = sqrt(f2) - sqrt(f1);
          return smoothstep(0.38, 0.04, edge);
        }

        void main() {
          vec2 uv = vUv * 5.8;
          float t = uTime * 0.55;

          vec2 warp1 = vec2(
            sin(uv.y * 1.3 + t * 0.9) * 0.20,
            cos(uv.x * 1.3 - t * 0.8) * 0.20
          );
          vec2 warp2 = vec2(
            cos(uv.x * 1.5 - t * 0.7) * 0.16,
            sin(uv.y * 1.4 + t * 0.85) * 0.16
          );

          float c1 = softVoronoiCaustic(uv + warp1, t);
          float c2 = softVoronoiCaustic(uv * 1.15 - warp2 + vec2(3.7, 1.9), -t * 0.85);
          float causticWeb = (c1 * 0.5 + c2 * 0.5);

          // Broad, soft diagonal sunbeams streaming gently from top-right
          float rayCoord = vUv.x * 0.78 - vUv.y * 0.62;
          float ray1 = 0.5 + 0.5 * sin(rayCoord * 11.0 + uTime * 0.35 + sin(vUv.y * 3.5) * 0.4);
          float ray2 = 0.5 + 0.5 * sin(rayCoord * 19.0 - uTime * 0.25 + cos(vUv.x * 4.0) * 0.35);
          float sunFade = smoothstep(0.0, 1.0, vUv.x * 0.6 + (1.0 - vUv.y) * 0.6);
          float lightRays = smoothstep(0.25, 0.95, ray1 * 0.6 + ray2 * 0.4) * sunFade;

          vec3 lightTint = vec3(0.76, 0.97, 0.99);
          float alpha = causticWeb * 0.105 + lightRays * 0.065;
          gl_FragColor = vec4(lightTint, clamp(alpha, 0.0, 0.20));
        }
      `
    });
    const causticMesh = new THREE.Mesh(new THREE.PlaneGeometry(14, 14), causticMat);
    causticMesh.rotation.x = -Math.PI / 2;
    causticMesh.position.y = -0.535;
    scene.add(causticMesh);

    // 3. Water ripple plane positioned at y = -0.51 so ripples render UNDER the 3D ducks, never over them
    const rippleMat = new THREE.MeshBasicMaterial({
      map: duckState.rippleTex,
      transparent: true,
      depthWrite: false
    });
    const rippleMesh = new THREE.Mesh(new THREE.PlaneGeometry(14, 14), rippleMat);
    rippleMesh.rotation.x = -Math.PI / 2;
    rippleMesh.position.y = -0.51;
    scene.add(rippleMesh);

    // Load media/rubber_duck.glb
    gltfLoader.load('media/rubber_duck.glb', (gltf) => {
      const rawScene = gltf.scene;
      rawScene.updateMatrixWorld(true);

      // Align the duck using the inverse of Cube.001_0's baked rotation so +Y is UP and +Z is BEAK
      const cubeNode = rawScene.getObjectByName('Cube.001_0');
      const alignQuat = new THREE.Quaternion();
      if (cubeNode) {
        cubeNode.getWorldQuaternion(alignQuat);
        alignQuat.invert();
      }

      const alignedHolder = new THREE.Group();
      alignedHolder.add(rawScene);
      rawScene.quaternion.premultiply(alignQuat);
      alignedHolder.updateMatrixWorld(true);

      // Compute exact bounding box & center at (0, 0, 0), normalized to unit radius
      const bbox = new THREE.Box3().setFromObject(alignedHolder);
      const center = new THREE.Vector3();
      const size = new THREE.Vector3();
      bbox.getCenter(center);
      bbox.getSize(size);

      const prototypeDuck = new THREE.Group();
      rawScene.position.sub(center);
      prototypeDuck.add(alignedHolder);

      const maxDim = Math.max(size.x, size.y, size.z) || 1;
      const normScale = 1.32 / maxDim;
      alignedHolder.scale.setScalar(normScale);

      prototypeDuck.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          if (child.material) {
            child.material.roughness = 0.24;
            child.material.metalness = 0.04;
          }
        }
      });

      // Register static 3D miniature duck for the side navigation thumbnail
      registerThumbnailScene(
        'ducks',
        prototypeDuck.clone(true),
        new THREE.Vector3(0, 0.72, 2.15),
        new THREE.Vector3(0, 0.02, 0),
        -0.68,
        0.08
      );

      // Spawn 30 ducks organically across the water tile: some upright, some lying flat on their side!
      const rand = mulberry32(7721);
      const count = 30;
      const bound = 3.85;

      // Golden-angle phyllotaxis seed + random jitter so there is zero grid look
      const goldenAngle = Math.PI * (3 - Math.sqrt(5));
      for (let i = 0; i < count; i++) {
        const duckMesh = prototypeDuck.clone(true);
        const rad = Math.sqrt((i + 0.5) / count) * bound * 0.94;
        const theta = i * goldenAngle + (rand() - 0.5) * 0.45;
        const gx = Math.cos(theta) * rad + (rand() - 0.5) * 0.50;
        const gz = Math.sin(theta) * rad + (rand() - 0.5) * 0.50;

        // ~35% of the ducks float lying flat on their side as requested!
        const isLyingFlat = i % 3 === 1;
        const sideSign = rand() > 0.5 ? 1 : -1;
        const scaleVar = 0.85 + rand() * 0.20;
        duckMesh.scale.setScalar(scaleVar);

        scene.add(duckMesh);

        duckState.ducks.push({
          mesh: duckMesh,
          x: clamp(gx, -bound, bound),
          z: clamp(gz, -bound, bound),
          vx: (rand() - 0.5) * 0.45,
          vz: (rand() - 0.5) * 0.45,
          yaw: rand() * Math.PI * 2,
          yawVel: (rand() - 0.5) * 0.4,
          isLyingFlat,
          baseRoll: isLyingFlat ? sideSign * 1.48 : 0, // ~85 deg on its side vs 0 deg upright
          basePitch: isLyingFlat ? 0.12 : 0,
          wobbleX: 0,
          wobbleZ: 0,
          wobbleVelX: 0,
          wobbleVelZ: 0,
          bobPhase: rand() * Math.PI * 2,
          radius: 0.50 * scaleVar
        });
      }

      // Run 65 relaxation steps so all 37 ducks settle naturally without overlaps inside the visible frame
      for (let step = 0; step < 65; step++) {
        for (let i = 0; i < duckState.ducks.length; i++) {
          for (let j = i + 1; j < duckState.ducks.length; j++) {
            const a = duckState.ducks[i];
            const b = duckState.ducks[j];
            const dx = b.x - a.x;
            const dz = b.z - a.z;
            const minDist = (a.radius + b.radius) * 1.04;
            const d = Math.hypot(dx, dz) || 0.001;
            if (d < minDist) {
              const push = (minDist - d) * 0.5;
              const nx = dx / d;
              const nz = dz / d;
              a.x = clamp(a.x - nx * push, -bound, bound);
              a.z = clamp(a.z - nz * push, -bound, bound);
              b.x = clamp(b.x + nx * push, -bound, bound);
              b.z = clamp(b.z + nz * push, -bound, bound);
            }
          }
        }
      }

      duckState.loaded = true;
    });
  }

  function updateAndRenderDucks(dt, timeSec) {
    duckState.causticUniforms.uTime.value = timeSec;

    const ducks = duckState.ducks;
    const bound = 3.38;

    // Raycast pointer onto water plane
    if (pointer.active) {
      duckState.raycaster.setFromCamera(
        { x: pointer.ndcX, y: pointer.ndcY },
        duckState.camera
      );
      duckState.raycaster.ray.intersectPlane(
        duckState.waterPlane,
        duckState.pointerWorld
      );

      // Spawn subtle, less intense water ripples mapped to the 3D water plane under the ducks
      const moveFromLast = Math.hypot(
        pointer.x - duckState.lastRippleX,
        pointer.y - duckState.lastRippleY
      );
      if (moveFromLast > 26 && duckState.ripples.length < 8) {
        duckState.lastRippleX = pointer.x;
        duckState.lastRippleY = pointer.y;
        const rx = (duckState.pointerWorld.x / 14 + 0.5) * 512;
        const ry = (duckState.pointerWorld.z / 14 + 0.5) * 512;
        const circleSize = 3;
        const maxSize = rippleSettings.maxSize;
        const animationSpeed = rippleSettings.animationSpeed;
        const startOpacity = 0.36;
        duckState.ripples.unshift({
          x: rx,
          y: ry,
          circleSize,
          maxSize,
          opacity: startOpacity,
          animationSpeed,
          opacityStep: (animationSpeed / (maxSize - circleSize)) * startOpacity * 0.55
        });
      }
    } else {
      duckState.pointerWorld.set(999, 0, 999);
    }

    const px = duckState.pointerWorld.x;
    const pz = duckState.pointerWorld.z;
    const influenceR = 1.95;

    // 1. Pointer water disturbance & buoyancy forces
    for (let i = 0; i < ducks.length; i++) {
      const d = ducks[i];
      d.mouseExcite = Math.max(0, (d.mouseExcite || 0) - dt * 0.32);

      if (pointer.active) {
        const dx = d.x - px;
        const dz = d.z - pz;
        const dist = Math.hypot(dx, dz);
        if (dist < influenceR && dist > 0.001) {
          const push = Math.pow(1 - dist / influenceR, 1.5) * 15.5;
          let nx = dx / dist;
          let nz = dz / dist;

          // If the duck is near an edge wall and the mouse is pushing it outward toward the frame,
          // deflect the push vector inward toward the center of the pond so edge ducks are always pushed away cleanly
          if (Math.abs(d.x) > bound - 0.48 && nx * d.x > 0) {
            nx = -Math.sign(d.x) * 0.72;
          }
          if (Math.abs(d.z) > bound - 0.48 && nz * d.z > 0) {
            nz = -Math.sign(d.z) * 0.72;
          }
          const nLen = Math.hypot(nx, nz) || 1;
          nx /= nLen;
          nz /= nLen;

          d.vx += nx * push * dt;
          d.vz += nz * push * dt;
          d.yawVel += (nx * pointer.vy - nz * pointer.vx) * 0.008 * dt;
          // Induce playful bathtub splash wobble
          d.wobbleVelX += nz * push * 0.28 * dt;
          d.wobbleVelZ -= nx * push * 0.28 * dt;

          if (dist < influenceR * 0.72) {
            d.mouseExcite = Math.max(d.mouseExcite, clamp(1 - dist / (influenceR * 0.72), 0.35, 1.0));
          }

          // Loudest pond quack when mouse comes in direct contact with the duck
          const contactR = d.radius + 0.58;
          if (dist < contactR && timeSec - (d.lastMouseQuack || 0) > 0.42) {
            d.lastMouseQuack = timeSec;
            d.mouseExcite = 1.0;
            const contactStrength = clamp(1 - dist / contactR, 0.25, 1.0);
            playPondQuack(true, contactStrength, 1.0);
          }
        }
      }

      // Gentle ambient water drift
      d.vx += Math.sin(timeSec * 0.7 + d.bobPhase) * 0.08 * dt;
      d.vz += Math.cos(timeSec * 0.6 + d.bobPhase * 1.3) * 0.08 * dt;

      // Water drag
      const damping = Math.exp(-2.1 * dt);
      d.vx *= damping;
      d.vz *= damping;
      d.yawVel *= Math.exp(-3.2 * dt);

      d.x += d.vx * dt;
      d.z += d.vz * dt;
      d.yaw += d.yawVel * dt;

      // Turn upright ducks gently toward their velocity vector when moving fast
      const speed = Math.hypot(d.vx, d.vz);
      if (!d.isLyingFlat && speed > 0.25) {
        const targetYaw = Math.atan2(d.vx, d.vz);
        let diff = targetYaw - d.yaw;
        while (diff > Math.PI) diff -= Math.PI * 2;
        while (diff < -Math.PI) diff += Math.PI * 2;
        d.yaw += diff * clamp(dt * 3.0, 0, 1);
      }

      // Soft wall bounce inside the square bathtub tile (plus residual quack if bouncing after mouse contact)
      let wallBounceSpeed = 0;
      if (d.x < -bound) {
        const hitSpeed = Math.abs(d.vx);
        d.x = -bound;
        d.vx = Math.max(hitSpeed * 0.75, 0.18);
        if (hitSpeed > 0.25) d.wobbleVelZ -= Math.min(hitSpeed * 0.6, 0.8);
        wallBounceSpeed = Math.max(wallBounceSpeed, hitSpeed);
      } else if (d.x > bound) {
        const hitSpeed = Math.abs(d.vx);
        d.x = bound;
        d.vx = -Math.max(hitSpeed * 0.75, 0.18);
        if (hitSpeed > 0.25) d.wobbleVelZ += Math.min(hitSpeed * 0.6, 0.8);
        wallBounceSpeed = Math.max(wallBounceSpeed, hitSpeed);
      }
      if (d.z < -bound) {
        const hitSpeed = Math.abs(d.vz);
        d.z = -bound;
        d.vz = Math.max(hitSpeed * 0.75, 0.18);
        if (hitSpeed > 0.25) d.wobbleVelX += Math.min(hitSpeed * 0.6, 0.8);
        wallBounceSpeed = Math.max(wallBounceSpeed, hitSpeed);
      } else if (d.z > bound) {
        const hitSpeed = Math.abs(d.vz);
        d.z = bound;
        d.vz = -Math.max(hitSpeed * 0.75, 0.18);
        if (hitSpeed > 0.25) d.wobbleVelX -= Math.min(hitSpeed * 0.6, 0.8);
        wallBounceSpeed = Math.max(wallBounceSpeed, hitSpeed);
      }

      if (
        wallBounceSpeed > 0.16 &&
        d.mouseExcite > 0.14 &&
        timeSec - (d.lastCollideQuack || 0) > 0.30
      ) {
        d.lastCollideQuack = timeSec;
        playPondQuack(false, clamp(wallBounceSpeed * 0.65, 0.18, 0.9), d.mouseExcite);
      }
    }

    // 2. Elastic Duck-to-Duck Collisions (bouncing against each other in the bathtub!)
    for (let iter = 0; iter < 2; iter++) {
      for (let i = 0; i < ducks.length; i++) {
        for (let j = i + 1; j < ducks.length; j++) {
          const a = ducks[i];
          const b = ducks[j];
          const dx = b.x - a.x;
          const dz = b.z - a.z;
          const minDist = a.radius + b.radius;
          const distSq = dx * dx + dz * dz;

          if (distSq < minDist * minDist && distSq > 0.00001) {
            const dist = Math.sqrt(distSq);
            const nx = dx / dist;
            const nz = dz / dist;
            const overlap = (minDist - dist) * 0.52;

            a.x = clamp(a.x - nx * overlap, -bound, bound);
            a.z = clamp(a.z - nz * overlap, -bound, bound);
            b.x = clamp(b.x + nx * overlap, -bound, bound);
            b.z = clamp(b.z + nz * overlap, -bound, bound);

            // Relative velocity along collision normal
            const rvx = b.vx - a.vx;
            const rvz = b.vz - a.vz;
            const velAlongNormal = rvx * nx + rvz * nz;

            if (velAlongNormal < -0.018) {
              const restitution = 0.78;
              const jMag = -(1 + restitution) * velAlongNormal * 0.5;

              a.vx -= nx * jMag;
              a.vz -= nz * jMag;
              b.vx += nx * jMag;
              b.vz += nz * jMag;

              // Propagate residual mouse-collision energy across chain-reacting ducks
              const sharedExcite = Math.max(a.mouseExcite || 0, b.mouseExcite || 0) * 0.88;
              a.mouseExcite = Math.max(a.mouseExcite || 0, sharedExcite);
              b.mouseExcite = Math.max(b.mouseExcite || 0, sharedExcite);

              if (jMag > 0.12) {
                const bumpTilt = clamp(jMag * 0.65, 0, 1.1);
                a.wobbleVelX -= nz * bumpTilt;
                a.wobbleVelZ += nx * bumpTilt;
                b.wobbleVelX += nz * bumpTilt;
                b.wobbleVelZ -= nx * bumpTilt;
                a.yawVel = clamp(a.yawVel + (Math.random() - 0.5) * bumpTilt * 0.9, -1.6, 1.6);
                b.yawVel = clamp(b.yawVel + (Math.random() - 0.5) * bumpTilt * 0.9, -1.6, 1.6);
              }

              // Trigger softer, fading-away residual pond quacks on duck-to-duck bounces
              const minImpact = sharedExcite > 0.10 ? 0.016 : 0.042;
              if (
                iter === 0 &&
                jMag > minImpact &&
                (timeSec - (a.lastCollideQuack || 0) > 0.26 ||
                  timeSec - (b.lastCollideQuack || 0) > 0.26)
              ) {
                a.lastCollideQuack = timeSec;
                b.lastCollideQuack = timeSec;
                const impactStrength = clamp((jMag - minImpact) * 1.15 + sharedExcite * 0.35, 0.18, 1.0);
                playPondQuack(false, impactStrength, sharedExcite);
              }
            }
          }
        }
      }
    }

    // 3. Spring-damper rocking wobble & mesh transform update
    for (let i = 0; i < ducks.length; i++) {
      const d = ducks[i];
      const springK = 22.0;
      const damp = 5.0;
      d.wobbleVelX = clamp(
        d.wobbleVelX + (-springK * d.wobbleX - damp * d.wobbleVelX) * dt,
        -3.2,
        3.2
      );
      d.wobbleVelZ = clamp(
        d.wobbleVelZ + (-springK * d.wobbleZ - damp * d.wobbleVelZ) * dt,
        -3.2,
        3.2
      );
      d.wobbleX = clamp(d.wobbleX + d.wobbleVelX * dt, -0.42, 0.42);
      d.wobbleZ = clamp(d.wobbleZ + d.wobbleVelZ * dt, -0.42, 0.42);

      const bobY =
        Math.sin(timeSec * 2.4 + d.bobPhase) * 0.055 +
        (d.isLyingFlat ? -0.1 : 0.02);
      const ambientRock = Math.cos(timeSec * 1.9 + d.bobPhase) * 0.045;

      d.mesh.position.set(d.x, bobY, d.z);
      d.mesh.rotation.order = 'YXZ';
      d.mesh.rotation.y = d.yaw;
      d.mesh.rotation.x = d.basePitch + d.wobbleX + ambientRock;
      d.mesh.rotation.z = d.baseRoll + d.wobbleZ + ambientRock * 0.8;
    }

    // 4. Update the 3D water ripple texture (rendered on the plane UNDER the ducks)
    const rCtx = duckState.rippleCanvas.getContext('2d');
    rCtx.clearRect(0, 0, 512, 512);
    if (duckState.ripples.length > 0) {
      rCtx.save();
      rCtx.filter = 'blur(4px)';
      rCtx.lineWidth = 2.2;
      const frameScale = clamp(dt * 45, 0.4, 2.0);
      const [sr, sg, sb] = rippleSettings.strokeColor;

      for (let i = duckState.ripples.length - 1; i >= 0; i--) {
        const rp = duckState.ripples[i];
        rp.circleSize += rp.animationSpeed * frameScale;
        rp.opacity -= rp.opacityStep * frameScale;

        if (rp.opacity <= 0) {
          duckState.ripples.splice(i, 1);
          continue;
        }

        rCtx.beginPath();
        rCtx.strokeStyle = `rgba(${sr}, ${sg}, ${sb}, ${rp.opacity})`;
        rCtx.arc(rp.x, rp.y, rp.circleSize, 0, Math.PI * 2);
        rCtx.stroke();
      }
      rCtx.restore();
    }
    duckState.rippleTex.needsUpdate = true;

    // Ensure 2D overlay canvas is clear so nothing draws over the 3D ducks
    ctx2d.clearRect(0, 0, canvas2d.width, canvas2d.height);

    // Render 3D scene (with ripples underneath the ducks)
    renderer.render(duckState.scene, duckState.camera);
  }

  // ============================================================================
  // EXHIBIT III: BUSY CROSS JUNCTION (Three.js + bus.glb, scooter.glb, Car.glb)
  // ============================================================================
  const junctionState = {
    scene: new THREE.Scene(),
    camera: new THREE.OrthographicCamera(-5.5, 5.5, 5.5, -5.5, 0.1, 60),
    vehicles: [],
    roadPlane: new THREE.Plane(new THREE.Vector3(0, 1, 0), 0),
    raycaster: new THREE.Raycaster(),
    pedestrianWorld: new THREE.Vector3(999, 0, 999),
    prevPedWorld: new THREE.Vector3(999, 0, 999),
    personGroup: null,
    personInner: null,
    personMixer: null,
    personAction: null,
    personYaw: 0,
    greenAxis: 0,
    greenTimer: 0,
    loadedCount: 0
  };

  // Curated vintage vehicle paint palette
  const CAR_COLORS = [
    '#c84638', // terracotta crimson
    '#d99b2e', // mustard ochre
    '#4d7a58', // botanical sage
    '#3b668c', // slate cobalt
    '#e8dec5', // warm cream ivory
    '#c96a36', // burnt tangerine
    '#52595e'  // charcoal slate
  ];

  function createIntersectionRoadTexture() {
    const c = document.createElement('canvas');
    c.width = 1024;
    c.height = 1024;
    const g = c.getContext('2d');
    const S = 1024;
    const center = S / 2;
    const roadHalf = S * 0.24; // On 12x12 world plane: 12 * 0.24 = 2.88 world units half-width

    // 1. Warm sandstone sidewalk base
    g.fillStyle = '#d8cbb4';
    g.fillRect(0, 0, S, S);

    // Sidewalk paving grid
    g.strokeStyle = 'rgba(90, 78, 60, 0.16)';
    g.lineWidth = 2;
    const tileStep = 64;
    for (let i = 0; i <= S; i += tileStep) {
      g.beginPath();
      g.moveTo(i, 0);
      g.lineTo(i, S);
      g.stroke();
      g.beginPath();
      g.moveTo(0, i);
      g.lineTo(S, i);
      g.stroke();
    }

    // 2. Asphalt roadways (N-S and E-W)
    g.fillStyle = '#3b3d3f';
    g.fillRect(center - roadHalf, 0, roadHalf * 2, S);
    g.fillRect(0, center - roadHalf, S, roadHalf * 2);

    // Curb borders
    g.strokeStyle = '#b7a88e';
    g.lineWidth = 8;
    const corners = [
      [0, 0, center - roadHalf, center - roadHalf],
      [center + roadHalf, 0, S - (center + roadHalf), center - roadHalf],
      [0, center + roadHalf, center - roadHalf, S - (center + roadHalf)],
      [center + roadHalf, center + roadHalf, S - (center + roadHalf), S - (center + roadHalf)]
    ];
    for (const [x, y, w, h] of corners) {
      g.strokeRect(x, y, w, h);
    }

    // 3. Double yellow/cream center dividers on all 4 arms
    g.strokeStyle = '#dfbe6f';
    g.lineWidth = 4;
    const stopOffset = roadHalf + 56;
    const arms = [
      [center, 0, center, center - stopOffset],
      [center, center + stopOffset, center, S],
      [0, center, center - stopOffset, center],
      [center + stopOffset, center, S, center]
    ];
    for (const [x0, y0, x1, y1] of arms) {
      const isVert = x0 === x1;
      for (const off of [-4, 4]) {
        g.beginPath();
        g.moveTo(x0 + (isVert ? off : 0), y0 + (isVert ? 0 : off));
        g.lineTo(x1 + (isVert ? off : 0), y1 + (isVert ? 0 : off));
        g.stroke();
      }
    }

    // 4. Dashed lane dividers (2 lanes per direction)
    g.strokeStyle = 'rgba(238, 230, 212, 0.65)';
    g.lineWidth = 3;
    g.setLineDash([22, 22]);
    const laneHalf = roadHalf * 0.5;
    for (const off of [-laneHalf, laneHalf]) {
      // N & S arms
      g.beginPath();
      g.moveTo(center + off, 0);
      g.lineTo(center + off, center - stopOffset);
      g.moveTo(center + off, center + stopOffset);
      g.lineTo(center + off, S);
      // E & W arms
      g.moveTo(0, center + off);
      g.lineTo(center - stopOffset, center + off);
      g.moveTo(center + stopOffset, center + off);
      g.lineTo(S, center + off);
      g.stroke();
    }
    g.setLineDash([]);

    // 5. Crisp Zebra Pedestrian Crosswalks on all 4 approaches
    g.fillStyle = '#eae1cd';
    const cwWidth = 40;
    const cwDist = roadHalf + 8;
    const stripeCount = 9;
    const stripeStep = (roadHalf * 1.84) / stripeCount;
    const stripeThick = stripeStep * 0.56;

    for (let i = 0; i < stripeCount; i++) {
      const pos = center - roadHalf * 0.92 + i * stripeStep + (stripeStep - stripeThick) * 0.5;
      // North crosswalk
      g.fillRect(pos, center - cwDist - cwWidth, stripeThick, cwWidth);
      // South crosswalk
      g.fillRect(pos, center + cwDist, stripeThick, cwWidth);
      // West crosswalk
      g.fillRect(center - cwDist - cwWidth, pos, cwWidth, stripeThick);
      // East crosswalk
      g.fillRect(center + cwDist, pos, cwWidth, stripeThick);
    }

    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    return tex;
  }

  function setupJunctionScene() {
    const { scene, camera } = junctionState;
    scene.background = new THREE.Color('#d8cbb4');

    // Slight isometric-bird's-eye angle so 3D buses, cars, and scooters show both roof and side profile!
    camera.position.set(0, 18, 3.8);
    camera.lookAt(0, 0, 0);

    const ambLight = new THREE.AmbientLight('#fff6e6', 1.45);
    scene.add(ambLight);

    const sunLight = new THREE.DirectionalLight('#fffbf0', 2.2);
    sunLight.position.set(8, 18, 9);
    sunLight.castShadow = true;
    sunLight.shadow.mapSize.set(1024, 1024);
    const d = 6.5;
    sunLight.shadow.camera.left = -d;
    sunLight.shadow.camera.right = d;
    sunLight.shadow.camera.top = d;
    sunLight.shadow.camera.bottom = -d;
    sunLight.shadow.bias = -0.0008;
    scene.add(sunLight);

    // Roadway ground plane (12x12 world units; roadHalf = 12 * 0.24 = 2.88)
    const roadGeo = new THREE.PlaneGeometry(12, 12);
    const roadMat = new THREE.MeshStandardMaterial({
      map: createIntersectionRoadTexture(),
      roughness: 0.85,
      metalness: 0.05
    });
    const roadMesh = new THREE.Mesh(roadGeo, roadMat);
    roadMesh.rotation.x = -Math.PI / 2;
    roadMesh.receiveShadow = true;
    scene.add(roadMesh);

    // Subtle sidewalk corner planters / trees for gallery miniature charm
    const cornerPositions = [
      [-4.35, -4.35],
      [4.35, -4.35],
      [-4.35, 4.35],
      [4.35, 4.35]
    ];
    for (const [cx, cz] of cornerPositions) {
      const crown = new THREE.Mesh(
        new THREE.SphereGeometry(0.62, 16, 12),
        new THREE.MeshStandardMaterial({ color: '#5c7c57', roughness: 0.8 })
      );
      crown.position.set(cx, 0.55, cz);
      crown.scale.set(1, 0.72, 1);
      crown.castShadow = true;
      crown.receiveShadow = true;
      scene.add(crown);
    }

    // Load media/person.glb as our top-down 3D pedestrian cursor
    gltfLoader.load('media/person.glb', (gltf) => {
      const personRoot = new THREE.Group();
      const personInner = new THREE.Group();
      const model = gltf.scene;

      model.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          child.frustumCulled = false;
          // 719 facial blend shapes are invisible from the top-down view but take seconds to upload on first render
          child.geometry.morphAttributes = {};
          child.morphTargetInfluences = undefined;
          child.morphTargetDictionary = undefined;
          if (child.material) {
            const mName = child.material.name || '';
            if (mName.includes('Hair') || mName.includes('Scalp')) {
              child.material.alphaTest = 0.25;
              child.material.depthWrite = true;
            }
          }
        }
      });

      let mixer = null;
      let action = null;
      if (gltf.animations && gltf.animations.length > 0) {
        const clip = gltf.animations[0];
        clip.tracks = clip.tracks.filter((track) => !track.name.endsWith('.morphTargetInfluences'));
        mixer = new THREE.AnimationMixer(model);
        action = mixer.clipAction(clip);
        action.play();
        // Advance to a natural mid-stride frame so arms & legs are relaxed and legible from top-down
        mixer.update(0.32);
      }

      // Center hip/head axis over (0, 0) with shoe soles resting at y = 0 (native height is 1.70 units)
      model.position.set(0, 0, 0.09);

      const hScale = 0.96 / 1.7;
      personInner.scale.set(hScale * 1.28, hScale, hScale * 1.28);
      personInner.add(model);

      personRoot.add(personInner);
      scene.add(personRoot);

      // Compile shaders and upload buffers now so the first pointer move doesn't freeze the page
      const warmTarget = new THREE.WebGLRenderTarget(16, 16);
      renderer.compileAsync(scene, camera).catch(() => { }).then(() => {
        renderer.setRenderTarget(warmTarget);
        renderer.render(scene, camera);
        renderer.setRenderTarget(null);
        warmTarget.dispose();
        personRoot.visible = false;

        junctionState.personGroup = personRoot;
        junctionState.personInner = personInner;
        junctionState.personMixer = mixer;
        junctionState.personAction = action;
      });
    });

    // Load the 3 vehicle GLB templates: bus, scooter, and Car
    const templates = {};

    function normalizeVehicleModel(gltfScene, targetLength, rotateY = 0) {
      const wrapper = new THREE.Group();
      const inner = new THREE.Group();
      inner.add(gltfScene);
      inner.rotation.y = rotateY;
      wrapper.add(inner);
      wrapper.updateMatrixWorld(true);

      const bbox = new THREE.Box3().setFromObject(wrapper);
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      bbox.getSize(size);
      bbox.getCenter(center);

      // Shift so bottom touches y = 0 and X/Z are centered at 0
      inner.position.x -= center.x;
      inner.position.z -= center.z;
      inner.position.y -= bbox.min.y;

      const scale = targetLength / (size.z || 1);
      wrapper.scale.setScalar(scale);

      wrapper.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
        }
      });

      return wrapper;
    }

    function checkAllVehiclesReady() {
      if (templates.car && templates.bus && templates.scooter) {
        spawnJunctionTraffic(templates);
      }
    }

    gltfLoader.load('media/vehicles/toy-car/source/Car.glb', (gltf) => {
      templates.car = normalizeVehicleModel(gltf.scene, 1.12, 0);

      // Register miniature 3D car for the 'Tourist in New York' side navigation thumbnail
      const thumbCar = templates.car.clone(true);
      thumbCar.traverse((child) => {
        if (child.isMesh && child.material && child.material.name === 'Car') {
          child.material = child.material.clone();
          child.material.color.set('#c84638');
        }
      });
      const thumbCarHolder = new THREE.Group();
      thumbCar.rotation.y = -1.05;
      thumbCarHolder.add(thumbCar);
      thumbCarHolder.updateMatrixWorld(true);
      const carBox = new THREE.Box3().setFromObject(thumbCarHolder);
      const carCenter = new THREE.Vector3();
      carBox.getCenter(carCenter);
      thumbCar.position.sub(carCenter);

      registerThumbnailScene(
        'junction',
        thumbCarHolder,
        new THREE.Vector3(0, 0.36, 2.02),
        new THREE.Vector3(0, 0, 0),
        0,
        0.02
      );

      checkAllVehiclesReady();
    });

    gltfLoader.load('media/vehicles/bus.glb', (gltf) => {
      // In bus.glb, front is along +X; rotating by -PI/2 around Y aligns front to +Z
      templates.bus = normalizeVehicleModel(gltf.scene, 1.82, -Math.PI / 2);
      checkAllVehiclesReady();
    });

    gltfLoader.load('media/vehicles/scooter.glb', (gltf) => {
      templates.scooter = normalizeVehicleModel(gltf.scene, 0.66, 0);
      checkAllVehiclesReady();
    });
  }

  function spawnJunctionTraffic(templates) {
    const { scene, vehicles } = junctionState;
    const rand = mulberry32(9042);

    // 8 traffic lanes (2 lanes in each of the 4 cardinal directions)
    // Road spans [-2.88, +2.88] in world units (width = 5.76, each lane = 1.44 wide)
    // Exact lane centers: ±0.72 (inner lanes) and ±2.16 (outer lanes)
    const LANES = [
      // Southbound (+Z): x = -2.16 (outer), x = -0.72 (inner)
      { id: 0, dirX: 0, dirZ: 1, yaw: 0, fixedX: -2.16, fixedZ: null, speed: 2.35 },
      { id: 1, dirX: 0, dirZ: 1, yaw: 0, fixedX: -0.72, fixedZ: null, speed: 2.65 },
      // Northbound (-Z): x = +0.72 (inner), x = +2.16 (outer)
      { id: 2, dirX: 0, dirZ: -1, yaw: Math.PI, fixedX: 0.72, fixedZ: null, speed: 2.55 },
      { id: 3, dirX: 0, dirZ: -1, yaw: Math.PI, fixedX: 2.16, fixedZ: null, speed: 2.25 },
      // Eastbound (+X): z = +0.72 (inner), z = +2.16 (outer)
      { id: 4, dirX: 1, dirZ: 0, yaw: Math.PI * 0.5, fixedX: null, fixedZ: 0.72, speed: 2.60 },
      { id: 5, dirX: 1, dirZ: 0, yaw: Math.PI * 0.5, fixedX: null, fixedZ: 2.16, speed: 2.30 },
      // Westbound (-X): z = -2.16 (outer), z = -0.72 (inner)
      { id: 6, dirX: -1, dirZ: 0, yaw: -Math.PI * 0.5, fixedX: null, fixedZ: -2.16, speed: 2.35 },
      { id: 7, dirX: -1, dirZ: 0, yaw: -Math.PI * 0.5, fixedX: null, fixedZ: -0.72, speed: 2.65 }
    ];

    let colorIdx = 0;

    for (let lIdx = 0; lIdx < LANES.length; lIdx++) {
      const lane = LANES[lIdx];
      const count = 3;
      const isEastWest = lane.fixedZ !== null;

      for (let k = 0; k < count; k++) {
        const roll = rand();
        let type = 'car';
        let length = 1.12;
        let width = 0.64;
        if (roll < 0.25 && k === 0) {
          type = 'bus';
          length = 1.82;
          width = 0.74;
        } else if (roll > 0.72) {
          type = 'scooter';
          length = 0.66;
          width = 0.38;
        }

        const mesh = templates[type].clone(true);

        // Tint individual cars with distinct colors from CAR_COLORS
        if (type === 'car') {
          const tintHex = CAR_COLORS[colorIdx++ % CAR_COLORS.length];
          mesh.traverse((child) => {
            if (child.isMesh && child.material && child.material.name === 'Car') {
              child.material = child.material.clone();
              child.material.color.set(tintHex);
            }
          });
        }

        // Add subtle red brake-light glow sprites at the rear bumper
        const brakeGroup = new THREE.Group();
        const brakeMat = new THREE.MeshBasicMaterial({ color: '#ff2a1a' });
        const brakeGeo = new THREE.SphereGeometry(0.068, 8, 8);
        const leftBrake = new THREE.Mesh(brakeGeo, brakeMat);
        const rightBrake = new THREE.Mesh(brakeGeo, brakeMat);
        const bw = type === 'scooter' ? 0.0 : type === 'bus' ? 0.25 : 0.22;
        leftBrake.position.set(-bw, 0.18, -length * 0.48);
        rightBrake.position.set(bw, 0.18, -length * 0.48);
        brakeGroup.add(leftBrake, rightBrake);
        brakeGroup.visible = false;
        mesh.add(brakeGroup);

        // Stagger initial positions so N-S starts crossing the box while E-W waits cleanly outside the box
        const baseS = !isEastWest
          ? (k === 0 ? -6.8 : k === 1 ? -1.2 : 3.8)
          : (k === 0 ? -7.4 : k === 1 ? -4.1 : 3.6);
        const sPos = baseS + (rand() - 0.5) * 0.35;
        const x = lane.fixedX !== null ? lane.fixedX : sPos * lane.dirX;
        const z = lane.fixedZ !== null ? lane.fixedZ : sPos * lane.dirZ;

        mesh.position.set(x, 0, z);
        mesh.rotation.y = lane.yaw;
        scene.add(mesh);

        vehicles.push({
          mesh,
          brakeGroup,
          lane,
          type,
          length,
          width,
          s: sPos, // signed progress along lane direction (-7.8 to +7.8)
          speed: isEastWest && k === 1 ? 0 : lane.speed,
          maxSpeed: lane.speed * (0.95 + rand() * 0.10),
          braking: false,
          stoppedByPed: false
        });
      }
    }
  }

  const cursorTargetWorld = new THREE.Vector3(999, 0, 999);
  const PERSON_BODY_RADIUS = 0.34;
  const CORNER_TREE_POSITIONS = [
    [-4.35, -4.35],
    [4.35, -4.35],
    [-4.35, 4.35],
    [4.35, 4.35]
  ];
  const TREE_CROWN_RADIUS = 0.64;

  /**
   * Pushes (px, pz) out of the 4 corner sidewalk trees (circles) and vehicles (smooth rounded rectangles).
   */
  function pushOutOfObstacles(px, pz) {
    const vehicles = junctionState.vehicles;
    const minTreeDist = TREE_CROWN_RADIUS + PERSON_BODY_RADIUS;
    let hit = false;
    let normX = 0;
    let normZ = 0;

    // 1. 4 corner sidewalk trees (circles)
    for (let t = 0; t < CORNER_TREE_POSITIONS.length; t++) {
      const [cx, cz] = CORNER_TREE_POSITIONS[t];
      const dx = px - cx;
      const dz = pz - cz;
      const dist = Math.hypot(dx, dz);
      if (dist < minTreeDist) {
        const nx = dist > 0.0001 ? dx / dist : (cx < 0 ? 1 : -1) * Math.SQRT1_2;
        const nz = dist > 0.0001 ? dz / dist : (cz < 0 ? 1 : -1) * Math.SQRT1_2;
        px = cx + nx * minTreeDist;
        pz = cz + nz * minTreeDist;
        hit = true;
        normX = nx;
        normZ = nz;
      }
    }

    // 2. Vehicles modeled as smooth rounded rectangles (inner hull + PERSON_BODY_RADIUS)
    for (let i = 0; i < vehicles.length; i++) {
      const v = vehicles[i];
      const lane = v.lane;
      const vx = lane.fixedX !== null ? lane.fixedX : v.s * lane.dirX;
      const vz = lane.fixedZ !== null ? lane.fixedZ : v.s * lane.dirZ;
      const bx = (lane.fixedX !== null ? v.width : v.length) * 0.5;
      const bz = (lane.fixedZ !== null ? v.width : v.length) * 0.5;

      const relX = px - vx;
      const relZ = pz - vz;
      const clX = clamp(relX, -bx, bx);
      const clZ = clamp(relZ, -bz, bz);
      const dx = relX - clX;
      const dz = relZ - clZ;
      const dist = Math.hypot(dx, dz);

      if (dist < PERSON_BODY_RADIUS) {
        let nx = 0;
        let nz = 0;
        if (dist > 0.0001) {
          nx = dx / dist;
          nz = dz / dist;
          px = vx + clX + nx * PERSON_BODY_RADIUS;
          pz = vz + clZ + nz * PERSON_BODY_RADIUS;
        } else {
          const penX = bx - Math.abs(relX);
          const penZ = bz - Math.abs(relZ);
          if (penX < penZ) {
            nx = relX >= 0 ? 1 : -1;
            px = vx + nx * (bx + PERSON_BODY_RADIUS);
          } else {
            nz = relZ >= 0 ? 1 : -1;
            pz = vz + nz * (bz + PERSON_BODY_RADIUS);
          }
        }
        hit = true;
        normX = nx;
        normZ = nz;
      }
    }

    return { x: px, z: pz, hit, normX, normZ };
  }

  /**
   * Advances the pedestrian from (px, pz) toward the cursor (tx, tz).
   * In open space, matches the cursor 1-for-1; when blocked by a vehicle or tree,
   * smoothly walks around the object's rounded perimeter over consecutive frames.
   */
  function stepPedestrianTowardCursor(px, pz, tx, tz, dt) {
    const totalDist = Math.hypot(tx - px, tz - pz);
    if (totalDist < 0.001) {
      const res = pushOutOfObstacles(px, pz);
      return { x: res.x, z: res.z };
    }

    const maxSubStep = 0.07;
    const steps = clamp(Math.ceil(totalDist / maxSubStep), 1, 24);
    const maxSlideBudget = 4.2 * dt;
    let slideUsed = 0;

    for (let s = 0; s < steps; s++) {
      const remX = tx - px;
      const remZ = tz - pz;
      const remDist = Math.hypot(remX, remZ);
      if (remDist < 0.001) break;

      const stepLen = Math.min(remDist, maxSubStep);
      const candX = px + (remX / remDist) * stepLen;
      const candZ = pz + (remZ / remDist) * stepLen;

      const res = pushOutOfObstacles(candX, candZ);
      px = res.x;
      pz = res.z;

      if (res.hit) {
        const toTargetX = tx - px;
        const toTargetZ = tz - pz;
        const intoObstacle = toTargetX * res.normX + toTargetZ * res.normZ;
        if (intoObstacle < 0) {
          const tanX = -res.normZ;
          const tanZ = res.normX;
          const tanProj = toTargetX * tanX + toTargetZ * tanZ;
          if (Math.abs(tanProj) > 0.035) {
            const slideDir = Math.sign(tanProj);
            const slideStep = Math.min(
              Math.abs(tanProj) - 0.02,
              maxSlideBudget - slideUsed,
              0.05
            );
            if (slideStep > 0.001) {
              const slid = pushOutOfObstacles(
                px + tanX * slideDir * slideStep,
                pz + tanZ * slideDir * slideStep
              );
              px = slid.x;
              pz = slid.z;
              slideUsed += slideStep;
            }
          }
          if (slideUsed >= maxSlideBudget || Math.abs(tanProj) <= 0.035) {
            break;
          }
        }
      }
    }

    return { x: px, z: pz };
  }

  function updateAndRenderJunction(dt) {
    const {
      vehicles,
      camera,
      raycaster,
      roadPlane,
      pedestrianWorld,
      prevPedWorld,
      personGroup,
      personInner,
      personMixer
    } = junctionState;

    if (pointer.active) {
      raycaster.setFromCamera({ x: pointer.ndcX, y: pointer.ndcY }, camera);
      raycaster.ray.intersectPlane(roadPlane, cursorTargetWorld);

      if (pedestrianWorld.x > 500) {
        const initRes = pushOutOfObstacles(cursorTargetWorld.x, cursorTargetWorld.z);
        pedestrianWorld.set(initRes.x, 0, initRes.z);
        prevPedWorld.copy(pedestrianWorld);
      } else {
        const nextPos = stepPedestrianTowardCursor(
          pedestrianWorld.x,
          pedestrianWorld.z,
          cursorTargetWorld.x,
          cursorTargetWorld.z,
          dt
        );
        pedestrianWorld.set(nextPos.x, 0, nextPos.z);
      }

      if (personGroup && personInner) {
        personGroup.visible = true;
        personGroup.position.set(pedestrianWorld.x, 0.01, pedestrianWorld.z);

        if (prevPedWorld.x < 500) {
          const dx = pedestrianWorld.x - prevPedWorld.x;
          const dz = pedestrianWorld.z - prevPedWorld.z;
          const moveDist = Math.hypot(dx, dz);
          if (moveDist > 0.008) {
            const targetYaw = Math.atan2(dx, dz);
            let diff = targetYaw - junctionState.personYaw;
            while (diff > Math.PI) diff -= Math.PI * 2;
            while (diff < -Math.PI) diff += Math.PI * 2;
            junctionState.personYaw += diff * clamp(dt * 14, 0, 1);
            personInner.rotation.y = junctionState.personYaw;
            if (personMixer) {
              personMixer.update(clamp(moveDist * 1.45, dt * 0.65, dt * 1.8));
            }
          }
        }
        prevPedWorld.copy(pedestrianWorld);
      }
    } else {
      pedestrianWorld.set(999, 0, 999);
      prevPedWorld.set(999, 0, 999);
      if (personGroup) {
        personGroup.visible = false;
      }
    }

    const pedX = pedestrianWorld.x;
    const pedZ = pedestrianWorld.z;
    const pedRadius = 0.85; // Pedestrian braking safety bubble around 3D person

    const trackHalfSpan = 7.8;
    const STOP_LINE = -2.92;  // Stop line before the crosswalk / intersection box
    const BOX_COMMIT = -2.65; // Point where a vehicle's front bumper has entered the intersection box
    const BOX_CLEAR = 2.65;   // Point where a vehicle's rear bumper has cleared the intersection box

    // Alternating N-S (axis 0) and E-W (axis 1) intersection right-of-way platooning
    // so perpendicular traffic waits at the stop-line outside the box instead of deadlocking in the center
    let nsMovingInBox = false;
    let ewMovingInBox = false;
    let nsApproaching = false;
    let ewApproaching = false;

    for (let i = 0; i < vehicles.length; i++) {
      const v = vehicles[i];
      const front = v.s + v.length * 0.5;
      const rear = v.s - v.length * 0.5;
      const isEW = v.lane.fixedZ !== null;
      if (front > BOX_COMMIT && rear < BOX_CLEAR) {
        if (!v.stoppedByPed && v.speed > 0.12) {
          if (isEW) ewMovingInBox = true;
          else nsMovingInBox = true;
        }
      } else if (front > -5.2 && front <= BOX_COMMIT) {
        if (isEW) ewApproaching = true;
        else nsApproaching = true;
      }
    }

    const perpMovingInBox = junctionState.greenAxis === 0 ? ewMovingInBox : nsMovingInBox;
    if (!perpMovingInBox) {
      junctionState.greenTimer += dt;
    }

    if (junctionState.greenAxis === 0) {
      if (
        (junctionState.greenTimer >= 3.2 && ewApproaching) ||
        (!nsMovingInBox && !nsApproaching && ewApproaching)
      ) {
        junctionState.greenAxis = 1;
        junctionState.greenTimer = 0;
      }
    } else {
      if (
        (junctionState.greenTimer >= 3.2 && nsApproaching) ||
        (!ewMovingInBox && !ewApproaching && nsApproaching)
      ) {
        junctionState.greenAxis = 0;
        junctionState.greenTimer = 0;
      }
    }

    // Update each vehicle's speed and position
    for (let i = 0; i < vehicles.length; i++) {
      const v = vehicles[i];
      const lane = v.lane;
      const isEW = lane.fixedZ !== null;
      const vAxis = isEW ? 1 : 0;
      const vx = lane.fixedX !== null ? lane.fixedX : v.s * lane.dirX;
      const vz = lane.fixedZ !== null ? lane.fixedZ : v.s * lane.dirZ;
      const vFront = v.s + v.length * 0.5;
      const vRear = v.s - v.length * 0.5;

      let mustStop = false;
      let targetSpeed = v.maxSpeed;
      const wasStoppedByPed = Boolean(v.stoppedByPed);
      v.stoppedByPed = false;
      let queuedBehindPedStop = false;

      // 1. Check if the 3D person is blocking this vehicle's path ahead
      if (pointer.active) {
        const toPedX = pedX - vx;
        const toPedZ = pedZ - vz;
        const forwardDist = toPedX * lane.dirX + toPedZ * lane.dirZ;
        const lateralDist = Math.abs(toPedX * -lane.dirZ + toPedZ * lane.dirX);

        const frontGap = forwardDist - v.length * 0.5;
        if (lateralDist < pedRadius && frontGap > -0.25 && frontGap < 2.1) {
          v.stoppedByPed = true;
          if (frontGap < 0.78) {
            mustStop = true;
            targetSpeed = 0;
          } else {
            targetSpeed = Math.min(targetSpeed, v.maxSpeed * ((frontGap - 0.78) / 1.32));
          }
        }
      }

      // 2. Check distance to the closest vehicle ahead in the SAME lane (queueing + "Don't Block the Box")
      let minGapAhead = 999;
      let aheadStoppedInBox = false;
      for (let j = 0; j < vehicles.length; j++) {
        if (i === j) continue;
        const other = vehicles[j];
        if (other.lane.id !== lane.id) continue;

        const ds = other.s - v.s;
        if (ds > 0) {
          const bumperGap = ds - (v.length * 0.5 + other.length * 0.5);
          if (bumperGap < minGapAhead) {
            minGapAhead = bumperGap;
            if (other.stoppedByPed && bumperGap < 1.35) {
              queuedBehindPedStop = true;
            }
          }
          const otherRear = other.s - other.length * 0.5;
          if (otherRear < 3.5 && other.speed < 0.6) {
            aheadStoppedInBox = true;
          }
        }
      }

      if (minGapAhead < 1.5) {
        if (minGapAhead < 0.45) {
          mustStop = true;
          targetSpeed = 0;
        } else {
          targetSpeed = Math.min(
            targetSpeed,
            v.maxSpeed * clamp((minGapAhead - 0.45) / 1.05, 0, 1)
          );
        }
      }

      // 3. Check if any perpendicular vehicle currently inside the box is crossing or straddling this lane
      let perpBlockingThisLane = false;
      for (let j = 0; j < vehicles.length; j++) {
        const other = vehicles[j];
        if (other.lane.dirX * lane.dirX + other.lane.dirZ * lane.dirZ !== 0) continue;
        const oFront = other.s + other.length * 0.5;
        const oRear = other.s - other.length * 0.5;
        if (oFront <= BOX_COMMIT || oRear >= BOX_CLEAR) continue;

        const ix = lane.fixedX !== null ? lane.fixedX : other.lane.fixedX;
        const iz = lane.fixedZ !== null ? lane.fixedZ : other.lane.fixedZ;
        const ox = other.lane.fixedX !== null ? other.lane.fixedX : other.s * other.lane.dirX;
        const oz = other.lane.fixedZ !== null ? other.lane.fixedZ : other.s * other.lane.dirZ;
        const oDistToCross = (ix - ox) * other.lane.dirX + (iz - oz) * other.lane.dirZ;

        if (oDistToCross + other.length * 0.5 > -0.42) {
          if (Math.abs(oDistToCross) < other.length * 0.5 + 0.45 || !other.stoppedByPed) {
            perpBlockingThisLane = true;
            break;
          }
        }
      }

      // 4. Stop-Line Gating BEFORE entering the intersection box ("Don't Block the Box")
      if (vFront > -4.8 && vFront <= BOX_COMMIT) {
        const perpAxisMoving = isEW ? nsMovingInBox : ewMovingInBox;
        let pedInLaneBox = false;
        if (pointer.active) {
          const laneCoord = lane.fixedX !== null ? lane.fixedX : lane.fixedZ;
          const pedLat = Math.abs((lane.fixedX !== null ? pedX : pedZ) - laneCoord);
          const pedS = pedX * lane.dirX + pedZ * lane.dirZ;
          if (pedLat < pedRadius && pedS > -2.6 && pedS < 3.4) {
            pedInLaneBox = true;
            v.stoppedByPed = true;
          }
        }

        if (
          vAxis !== junctionState.greenAxis ||
          perpAxisMoving ||
          perpBlockingThisLane ||
          aheadStoppedInBox ||
          pedInLaneBox
        ) {
          const distToStop = STOP_LINE - vFront;
          if (distToStop <= 0.12) {
            mustStop = true;
            targetSpeed = 0;
          } else if (distToStop < 1.6) {
            targetSpeed = Math.min(
              targetSpeed,
              v.maxSpeed * clamp((distToStop - 0.12) / 1.48, 0, 1)
            );
          }
        }
      } else if (vFront > BOX_COMMIT && vRear < BOX_CLEAR) {
        // Secondary safety check for vehicles already inside the box if a perpendicular car is straddling ahead
        for (let j = 0; j < vehicles.length; j++) {
          const other = vehicles[j];
          if (other.lane.dirX * lane.dirX + other.lane.dirZ * lane.dirZ !== 0) continue;
          const oFront = other.s + other.length * 0.5;
          const oRear = other.s - other.length * 0.5;
          if (oFront <= BOX_COMMIT || oRear >= BOX_CLEAR) continue;

          const ix = lane.fixedX !== null ? lane.fixedX : other.lane.fixedX;
          const iz = lane.fixedZ !== null ? lane.fixedZ : other.lane.fixedZ;
          const ox = other.lane.fixedX !== null ? other.lane.fixedX : other.s * other.lane.dirX;
          const oz = other.lane.fixedZ !== null ? other.lane.fixedZ : other.s * other.lane.dirZ;

          const vDistToCross = (ix - vx) * lane.dirX + (iz - vz) * lane.dirZ;
          const oDistToCross = (ix - ox) * other.lane.dirX + (iz - oz) * other.lane.dirZ;
          const vFrontToCross = vDistToCross - v.length * 0.5;
          const oFrontToCross = oDistToCross - other.length * 0.5;

          if (vFrontToCross > -0.32 && vFrontToCross < 1.6 && oDistToCross + other.length * 0.5 > -0.40) {
            const otherStraddling = Math.abs(oDistToCross) < other.length * 0.5 + 0.36;
            const otherMovingCloser =
              !other.stoppedByPed &&
              other.speed > 0.12 &&
              oFrontToCross < vFrontToCross - 0.02;

            if (otherStraddling || otherMovingCloser) {
              if (vFrontToCross < 0.48) {
                mustStop = true;
                targetSpeed = 0;
                break;
              } else {
                targetSpeed = Math.min(
                  targetSpeed,
                  v.maxSpeed * clamp((vFrontToCross - 0.48) / 1.12, 0, 1)
                );
              }
            }
          }
        }
      }

      // Smooth acceleration / braking
      const accelRate = targetSpeed < v.speed ? 10.5 : 4.0;
      v.speed = lerp(v.speed, targetSpeed, clamp(dt * accelRate, 0, 1));
      if (mustStop && v.speed < 0.08) v.speed = 0;

      // 5. Hard physical 2D bounding-box collision prevention against other vehicles & pedestrian
      const nextS = v.s + v.speed * dt;
      const candX = lane.fixedX !== null ? lane.fixedX : nextS * lane.dirX;
      const candZ = lane.fixedZ !== null ? lane.fixedZ : nextS * lane.dirZ;
      const halfX = (lane.fixedX !== null ? v.width : v.length) * 0.5;
      const halfZ = (lane.fixedZ !== null ? v.width : v.length) * 0.5;

      let blockedPhysically = false;
      if (pointer.active) {
        const pedAhead = (pedX - vx) * lane.dirX + (pedZ - vz) * lane.dirZ > -v.length * 0.25;
        if (
          pedAhead &&
          Math.abs(candX - pedX) < halfX + PERSON_BODY_RADIUS + 0.06 &&
          Math.abs(candZ - pedZ) < halfZ + PERSON_BODY_RADIUS + 0.06
        ) {
          blockedPhysically = true;
          v.stoppedByPed = true;
        }
      }

      if (!blockedPhysically) {
        for (let j = 0; j < vehicles.length; j++) {
          if (i === j) continue;
          const other = vehicles[j];
          const oLane = other.lane;
          const ox = oLane.fixedX !== null ? oLane.fixedX : other.s * oLane.dirX;
          const oz = oLane.fixedZ !== null ? oLane.fixedZ : other.s * oLane.dirZ;
          const otherAhead = (ox - vx) * lane.dirX + (oz - vz) * lane.dirZ > 0.05;
          if (!otherAhead) continue;

          const oHalfX = (oLane.fixedX !== null ? other.width : other.length) * 0.5;
          const oHalfZ = (oLane.fixedZ !== null ? other.width : other.length) * 0.5;
          if (
            Math.abs(candX - ox) < halfX + oHalfX + 0.14 &&
            Math.abs(candZ - oz) < halfZ + oHalfZ + 0.14
          ) {
            blockedPhysically = true;
            break;
          }
        }
      }

      if (blockedPhysically) {
        v.speed = 0;
      } else {
        v.s = nextS;
      }

      v.braking = targetSpeed < v.maxSpeed * 0.65 || v.speed < 0.3;
      v.brakeGroup.visible = v.braking;

      // Trigger layered street honk (honk1 / honk2) when pedestrian causes a vehicle (or queue behind it) to stop/brake
      if (pointer.active && (v.stoppedByPed || (queuedBehindPedStop && v.braking))) {
        const nowSec = performance.now() * 0.001;
        if (!v.nextHonkTime) {
          v.nextHonkTime = nowSec + (wasStoppedByPed ? 1.6 : 0.04 + Math.random() * 0.25);
        }
        if (nowSec >= v.nextHonkTime && v.speed < v.maxSpeed * 0.55) {
          const level = v.stoppedByPed ? 0.85 + Math.random() * 0.35 : 0.45 + Math.random() * 0.35;
          playStreetHonk(level);
          v.nextHonkTime = nowSec + 1.85 + Math.random() * 1.65;
        }
      } else {
        v.nextHonkTime = 0;
      }

      // Wrap around cleanly behind the furthest-back vehicle in the same lane
      if (v.s > trackHalfSpan) {
        let minSInLane = trackHalfSpan;
        for (let j = 0; j < vehicles.length; j++) {
          if (j !== i && vehicles[j].lane.id === lane.id) {
            if (vehicles[j].s < minSInLane) {
              minSInLane = vehicles[j].s;
            }
          }
        }
        v.s = Math.min(-trackHalfSpan, minSInLane - v.length - 1.25);
      }

      const nx = lane.fixedX !== null ? lane.fixedX : v.s * lane.dirX;
      const nz = lane.fixedZ !== null ? lane.fixedZ : v.s * lane.dirZ;
      v.mesh.position.set(nx, 0, nz);
    }

    // Final post-vehicle-step check so pedestrian and vehicles never overlap
    if (pointer.active && personGroup) {
      const finalPed = pushOutOfObstacles(pedestrianWorld.x, pedestrianWorld.z);
      pedestrianWorld.set(finalPed.x, 0, finalPed.z);
      personGroup.position.set(finalPed.x, 0.01, finalPed.z);
    }

    renderer.render(junctionState.scene, junctionState.camera);
    ctx2d.clearRect(0, 0, canvas2d.width, canvas2d.height);
  }

  // ============================================================================
  // EXHIBIT IV: COMEDIAN — BANANA DUCT-TAPED TO WALL + EGGPLANT GLASS BLOB LENS
  // (Three.js + media/banana.glb + media/eggplant.glb + Warped Glass Blob Lens)
  // ============================================================================
  const BANANA_OBJECT_SCALE = 1.05;
  const bananaState = {
    scene: new THREE.Scene(),
    camera: new THREE.PerspectiveCamera(32, 1, 0.1, 50),
    rootGroup: new THREE.Group(),
    normalMeshes: [],
    xrayGroup: new THREE.Group(),
    loaded: false,
    lensRadius: 62,
    smoothX: 240,
    smoothY: 240,
    wasActive: false,
    lensBufferCanvas: document.createElement('canvas'),
    lensPatchCanvas: document.createElement('canvas')
  };

  function setupBananaScene() {
    const { scene, camera, rootGroup, xrayGroup } = bananaState;
    scene.background = new THREE.Color('#e8dec9');

    camera.position.set(0, 0, 3.15);
    camera.lookAt(0, 0, 0);

    const ambLight = new THREE.AmbientLight('#fff8eb', 1.65);
    scene.add(ambLight);

    const keyLight = new THREE.DirectionalLight('#fffdf8', 2.2);
    keyLight.position.set(2.2, 3.2, 4.0);
    keyLight.castShadow = true;
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight('#f2e2c4', 0.95);
    fillLight.position.set(-2.5, -1.5, 2.5);
    scene.add(fillLight);

    scene.add(rootGroup);

    gltfLoader.load('media/banana.glb', (gltf) => {
      const model = gltf.scene;

      // In banana.glb, the wall frame lies in X-Z and faces +Y; rotating X by +PI/2 faces +Z (the camera)
      const orientGroup = new THREE.Group();
      orientGroup.rotation.x = Math.PI / 2;
      orientGroup.add(model);
      orientGroup.updateMatrixWorld(true);

      const bbox = new THREE.Box3().setFromObject(orientGroup);
      const center = new THREE.Vector3();
      bbox.getCenter(center);
      orientGroup.position.sub(center);
      orientGroup.updateMatrixWorld(true);

      rootGroup.add(orientGroup);
      rootGroup.add(xrayGroup);
      xrayGroup.visible = false;

      let bananaMesh = null;
      let tapeMesh = null;
      let frameMesh = null;

      orientGroup.traverse((child) => {
        if (child.isMesh) {
          bananaState.normalMeshes.push(child);
          const matName = child.material ? child.material.name : '';
          if (matName === 'banana') bananaMesh = child;
          else if (matName === 'tape') tapeMesh = child;
          else if (matName === 'frame') frameMesh = child;
        }
      });

      // Register static 3D miniature of the banana + duct tape (without the wall background frame) for side thumbnails
      if (bananaMesh) {
        const thumbGroup = new THREE.Group();
        const bClone = bananaMesh.clone();
        bClone.applyMatrix4(bananaMesh.matrixWorld);
        thumbGroup.add(bClone);
        if (tapeMesh) {
          const tClone = tapeMesh.clone();
          tClone.applyMatrix4(tapeMesh.matrixWorld);
          thumbGroup.add(tClone);
        }
        const tBox = new THREE.Box3().setFromObject(thumbGroup);
        const tCenter = new THREE.Vector3();
        const tSize = new THREE.Vector3();
        tBox.getCenter(tCenter);
        tBox.getSize(tSize);
        const tMax = Math.max(tSize.x, tSize.y, tSize.z) || 1;
        const tScale = 1.25 / tMax;
        const thumbWrapper = new THREE.Group();
        thumbGroup.position.sub(tCenter);
        thumbWrapper.scale.setScalar(tScale);
        thumbWrapper.add(thumbGroup);

        registerThumbnailScene(
          'banana',
          thumbWrapper,
          new THREE.Vector3(0, 0, 2.25),
          new THREE.Vector3(0, 0, 0),
          0.18,
          0.08
        );
      }

      // Build the hidden 3D Eggplant surprise aligned behind the banana and under the duct tape
      if (bananaMesh) {
        buildBananaEggplantSurprise(bananaMesh, tapeMesh, frameMesh);
      }

      // Scale the banana and silver tape up proportionally within the frame (matching xrayFruitGroup)
      const wallZ = frameMesh ? new THREE.Box3().setFromObject(frameMesh).max.z : 0;
      const normalFruitGroup = new THREE.Group();
      normalFruitGroup.position.set(0, 0, wallZ * (1 - BANANA_OBJECT_SCALE));
      normalFruitGroup.scale.setScalar(BANANA_OBJECT_SCALE);
      rootGroup.add(normalFruitGroup);

      [bananaMesh, tapeMesh].forEach((m) => {
        if (!m) return;
        m.updateMatrixWorld(true);
        const worldMat = m.matrixWorld.clone();
        if (m.parent) m.parent.remove(m);
        m.position.set(0, 0, 0);
        m.rotation.set(0, 0, 0);
        m.scale.set(1, 1, 1);
        m.updateMatrix();
        m.applyMatrix4(worldMat);
        normalFruitGroup.add(m);
      });

      bananaState.loaded = true;
    });
  }

  /**
   * Places media/eggplant.glb in the hidden xrayGroup aligned to the banana's diagonal arc
   * and strapped beneath the duct tape so hovering the magnifying glass reveals the eggplant instead.
   */
  function buildBananaEggplantSurprise(bananaMesh, tapeMesh, frameMesh) {
    const { xrayGroup } = bananaState;
    bananaMesh.parent.updateMatrixWorld(true);

    // 1. Keep the gallery wall frame identical inside the glass blob so only the banana transforms into the eggplant
    if (frameMesh) {
      const frameClone = frameMesh.clone();
      frameClone.applyMatrix4(frameMesh.matrixWorld);
      frameClone.material = frameMesh.material.clone();
      frameClone.renderOrder = 0;
      xrayGroup.add(frameClone);
    }

    const wallZ = frameMesh ? new THREE.Box3().setFromObject(frameMesh).max.z : 0;
    const xrayFruitGroup = new THREE.Group();
    xrayFruitGroup.position.set(0, 0, wallZ * (1 - BANANA_OBJECT_SCALE));
    xrayFruitGroup.scale.setScalar(BANANA_OBJECT_SCALE);
    xrayGroup.add(xrayFruitGroup);

    // 2. Replace the banana's silver duct tape in xrayGroup with the top-most 3D bandaid from media/bandaid_set.glb
    if (tapeMesh) {
      const geom = tapeMesh.geometry.clone();
      geom.applyMatrix4(tapeMesh.matrixWorld);

      // Compute the tape strip's principal length (u) and width (v) axes and 3D arch profile z(u)
      const pos = geom.attributes.position;
      let cx = 0;
      let cy = 0;
      for (let i = 0; i < pos.count; i++) {
        cx += pos.getX(i);
        cy += pos.getY(i);
      }
      cx /= pos.count || 1;
      cy /= pos.count || 1;

      let cxx = 0;
      let cxy = 0;
      let cyy = 0;
      for (let i = 0; i < pos.count; i++) {
        const dx = pos.getX(i) - cx;
        const dy = pos.getY(i) - cy;
        cxx += dx * dx;
        cxy += dx * dy;
        cyy += dy * dy;
      }
      const theta = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
      const ax = Math.cos(theta);
      const ay = Math.sin(theta);
      const bx = -ay;
      const by = ax;

      let minU = Infinity;
      let maxU = -Infinity;
      let minV = Infinity;
      let maxV = -Infinity;
      for (let i = 0; i < pos.count; i++) {
        const dx = pos.getX(i) - cx;
        const dy = pos.getY(i) - cy;
        const uProj = dx * ax + dy * ay;
        const vProj = dx * bx + dy * by;
        if (uProj < minU) minU = uProj;
        if (uProj > maxU) maxU = uProj;
        if (vProj < minV) minV = vProj;
        if (vProj > maxV) maxV = vProj;
      }
      const spanU = maxU - minU || 1;
      const spanV = maxV - minV || 1;
      const midU = 0.5 * (minU + maxU);
      const halfU = 0.5 * spanU || 1;

      // Sample the 3D arch height z(uNorm) of the original tape over the fruit
      const NUM_BINS = 24;
      const rawBins = new Float32Array(NUM_BINS).fill(-Infinity);
      for (let i = 0; i < pos.count; i++) {
        const dx = pos.getX(i) - cx;
        const dy = pos.getY(i) - cy;
        const uNorm = clamp((dx * ax + dy * ay - midU) / halfU, -1, 1);
        const binIdx = clamp(Math.floor(((uNorm + 1) * 0.5) * NUM_BINS), 0, NUM_BINS - 1);
        const pz = pos.getZ(i);
        if (pz > rawBins[binIdx]) rawBins[binIdx] = pz;
      }
      for (let b = 0; b < NUM_BINS; b++) {
        if (rawBins[b] === -Infinity) rawBins[b] = wallZ;
      }
      const smoothBins = new Float32Array(NUM_BINS);
      for (let b = 0; b < NUM_BINS; b++) {
        const prev = rawBins[Math.max(0, b - 1)];
        const curr = rawBins[b];
        const next = rawBins[Math.min(NUM_BINS - 1, b + 1)];
        smoothBins[b] = prev * 0.25 + curr * 0.5 + next * 0.25;
      }
      function sampleTapeArchZ(uNorm) {
        const t = clamp((uNorm + 1) * 0.5, 0, 1) * (NUM_BINS - 1);
        const idx0 = Math.floor(t);
        const idx1 = Math.min(NUM_BINS - 1, idx0 + 1);
        return lerp(smoothBins[idx0], smoothBins[idx1], t - idx0);
      }

      gltfLoader.load('media/bandaid_set.glb', (bGltf) => {
        let topBandaidMesh = null;
        let highestCenterX = -Infinity;

        bGltf.scene.traverse((child) => {
          if (child.isMesh && child.geometry) {
            child.geometry.computeBoundingBox();
            const bb = child.geometry.boundingBox;
            const centerX = 0.5 * (bb.min.x + bb.max.x);
            if (centerX > highestCenterX) {
              highestCenterX = centerX;
              topBandaidMesh = child;
            }
          }
        });

        if (!topBandaidMesh) return;

        const bGeom = topBandaidMesh.geometry.clone();
        bGeom.computeBoundingBox();
        const bb = bGeom.boundingBox;
        const midBx = 0.5 * (bb.min.x + bb.max.x);
        const halfBx = 0.5 * (bb.max.x - bb.min.x) || 1;
        const midBz = 0.5 * (bb.min.z + bb.max.z);
        const halfBz = 0.5 * (bb.max.z - bb.min.z) || 1;
        const minBy = bb.min.y;

        const bPos = bGeom.attributes.position;
        const bNorm = bGeom.attributes.normal;
        const bUv = bGeom.attributes.uv;
        const bIdx = bGeom.getIndex();

        // Filter out the underside grey gauze pad (ny <= 0.2 or py <= 0.022) and sort front triangles
        // so the raised center cushion pad (avgV > 0.72) always draws cleanly on top of the main strip
        if (bIdx && bNorm && bUv) {
          const indices = bIdx.array;
          const frontTris = [];
          for (let t = 0; t < indices.length; t += 3) {
            const i0 = indices[t];
            const i1 = indices[t + 1];
            const i2 = indices[t + 2];
            const py = (bPos.getY(i0) + bPos.getY(i1) + bPos.getY(i2)) / 3;
            const ny = (bNorm.getY(i0) + bNorm.getY(i1) + bNorm.getY(i2)) / 3;
            const avgV = (bUv.getY(i0) + bUv.getY(i1) + bUv.getY(i2)) / 3;

            if (ny > 0.2 && py > 0.022) {
              const isCenterPad = avgV > 0.72 ? 1 : 0;
              frontTris.push({ i0, i1, i2, isCenterPad, py });
            }
          }
          frontTris.sort((a, b) => {
            if (a.isCenterPad !== b.isCenterPad) return a.isCenterPad - b.isCenterPad;
            return a.py - b.py;
          });
          const sortedIndices = new Uint16Array(frontTris.length * 3);
          for (let k = 0; k < frontTris.length; k++) {
            sortedIndices[k * 3] = frontTris[k].i0;
            sortedIndices[k * 3 + 1] = frontTris[k].i1;
            sortedIndices[k * 3 + 2] = frontTris[k].i2;
          }
          bGeom.setIndex(new THREE.BufferAttribute(sortedIndices, 1));
        }

        for (let i = 0; i < bPos.count; i++) {
          const vx = bPos.getX(i);
          const vy = bPos.getY(i);
          const vz = bPos.getZ(i);

          const uNorm = (vz - midBz) / halfBz;
          const vNorm = (vx - midBx) / halfBx;

          const uDist = uNorm * (spanU * 0.52);
          const vDist = vNorm * (spanV * 0.38);

          const wx = cx + ax * uDist + bx * vDist;
          const wy = cy + ay * uDist + by * vDist;
          const wz = sampleTapeArchZ(uNorm) + (vy - minBy) * 1.35 + 0.006;

          bPos.setXYZ(i, wx, wy, wz);
        }
        bGeom.computeVertexNormals();

        const bMat = topBandaidMesh.material.clone();
        bMat.roughness = 0.55;
        bMat.metalness = 0.02;
        bMat.depthTest = false;
        bMat.depthWrite = false;

        const bandaidMesh = new THREE.Mesh(bGeom, bMat);
        bandaidMesh.receiveShadow = false;
        bandaidMesh.castShadow = false;
        bandaidMesh.renderOrder = 10;
        xrayFruitGroup.add(bandaidMesh);
      });
    }

    // 3. Load media/eggplant.glb and scale/curve it so its top & bottom tips reach closer to the banana's tips
    gltfLoader.load('media/eggplant.glb', (gltf) => {
      const eggplantModel = gltf.scene;
      eggplantModel.updateMatrixWorld(true);

      // Curve the eggplant's geometry along its vertical spine so its top stem & bottom bulb
      // sweep toward the banana's crescent tips while remaining a glossy purple eggplant
      eggplantModel.traverse((child) => {
        if (child.isMesh && child.geometry) {
          child.geometry = child.geometry.clone();
          const pos = child.geometry.attributes.position;
          const v = new THREE.Vector3();
          for (let i = 0; i < pos.count; i++) {
            v.fromBufferAttribute(pos, i);
            // In local space, Y runs along the eggplant length (~[-0.114, +0.114]); slightly slim the bottom half before bending along X
            const ny = v.y / 0.114;
            if (ny < 0.1) {
              const bottomT = clamp((0.1 - ny) / 1.1, 0, 1);
              const widthScale = 1.0 - 0.22 * (bottomT * bottomT * (3 - 2 * bottomT));
              v.x *= widthScale;
              v.z *= widthScale;
            }
            const bendWeight = ny < 0 ? 0.028 : 0.018;
            v.x += (ny * ny - 0.26) * bendWeight;
            pos.setXYZ(i, v.x, v.y, v.z);
          }
          child.geometry.computeVertexNormals();

          child.renderOrder = 1;
          child.receiveShadow = false;

          if (child.material) {
            child.material = child.material.clone();
            const matName = (child.material.name || '').toLowerCase();
            if (matName.includes('body')) {
              child.material.roughness = 0.22;
              child.material.metalness = 0.05;
            } else if (matName.includes('top')) {
              child.material.roughness = 0.48;
            }
          }
        }
      });

      const eggplantPivot = new THREE.Group();
      // Enlarged length & girth so top calyx and bottom bulb reach closer to the banana's top and bottom tips
      eggplantModel.scale.set(4.15, 5.05, 1.15);
      eggplantPivot.add(eggplantModel);

      // Align with the banana's diagonal arc (top stem toward upper-left tip, bottom bulb toward lower-right tip)
      eggplantPivot.position.set(0.015, 0.035, -0.012);
      eggplantPivot.rotation.z = 0.42;
      eggplantPivot.rotation.y = -0.10;

      xrayFruitGroup.add(eggplantPivot);
    });
  }

  function updateAndRenderBanana(dt, timeSec) {
    const {
      scene,
      camera,
      rootGroup,
      normalMeshes,
      xrayGroup,
      lensBufferCanvas,
      lensPatchCanvas
    } = bananaState;

    // Keep the wall plane completely flat (no tilt when hovering the canvas)
    rootGroup.rotation.set(0, 0, 0);

    if (pointer.active) {
      if (!bananaState.wasActive) {
        bananaState.smoothX = pointer.x;
        bananaState.smoothY = pointer.y;
        bananaState.wasActive = true;
      } else {
        bananaState.smoothX = lerp(bananaState.smoothX, pointer.x, clamp(dt * 28, 0, 1));
        bananaState.smoothY = lerp(bananaState.smoothY, pointer.y, clamp(dt * 28, 0, 1));
      }
    } else {
      bananaState.wasActive = false;
    }

    ctx2d.clearRect(0, 0, canvas2d.width, canvas2d.height);

    // Render the Dual-Pass Circular Warped Glass Blob only while the cursor is inside the canvas
    if (bananaState.loaded && pointer.active && pointer.haloAlpha > 0.01) {
      const lx = bananaState.smoothX;
      const ly = bananaState.smoothY;
      const isMobileView = window.innerWidth <= 680;
      const baseLensRadius = isMobileView
        ? Math.max(58, Math.round(tileSize * 0.195))
        : Math.max(bananaState.lensRadius, Math.round(tileSize * 0.132));
      const r = baseLensRadius * (0.90 + 0.10 * pointer.haloAlpha);

      // PASS 1: Render the hidden Eggplant + brown tape scene (xrayGroup)
      for (const m of normalMeshes) m.visible = false;
      xrayGroup.visible = true;
      camera.zoom = 1.0;
      camera.updateProjectionMatrix();
      renderer.render(scene, camera);

      const W = canvas2d.width;
      const H = canvas2d.height;
      if (lensBufferCanvas.width !== W || lensBufferCanvas.height !== H) {
        lensBufferCanvas.width = W;
        lensBufferCanvas.height = H;
      }
      const bCtx = lensBufferCanvas.getContext('2d', { willReadFrequently: true });
      bCtx.clearRect(0, 0, W, H);
      bCtx.drawImage(webglCanvas, 0, 0);

      const cx = lx * dpr;
      const cy = ly * dpr;
      const cr = r * dpr;
      const halfSpan = Math.ceil(cr + 2);
      const patchSize = halfSpan * 2;

      if (lensPatchCanvas.width !== patchSize || lensPatchCanvas.height !== patchSize) {
        lensPatchCanvas.width = patchSize;
        lensPatchCanvas.height = patchSize;
      }
      const pCtx = lensPatchCanvas.getContext('2d');

      // Read bounded source sub-rectangle around the glass lens
      const margin = 10;
      const rx0 = clamp(Math.floor(cx - halfSpan - margin), 0, W - 1);
      const ry0 = clamp(Math.floor(cy - halfSpan - margin), 0, H - 1);
      const rx1 = clamp(Math.ceil(cx + halfSpan + margin), rx0 + 1, W);
      const ry1 = clamp(Math.ceil(cy + halfSpan + margin), ry0 + 1, H);
      const rw = rx1 - rx0;
      const rh = ry1 - ry0;

      const srcImg = bCtx.getImageData(rx0, ry0, rw, rh);
      const src = srcImg.data;
      const dstImg = pCtx.createImageData(patchSize, patchSize);
      const dst = dstImg.data;

      function sampleBilinear(sx, sy, chOffset) {
        const x = clamp(sx - rx0, 0, rw - 1.001);
        const y = clamp(sy - ry0, 0, rh - 1.001);
        const x0 = x | 0;
        const y0 = y | 0;
        const fx = x - x0;
        const fy = y - y0;
        const row0 = y0 * rw;
        const row1 = (y0 + 1) * rw;
        const i00 = (row0 + x0) * 4 + chOffset;
        const i10 = i00 + 4;
        const i01 = (row1 + x0) * 4 + chOffset;
        const i11 = i01 + 4;
        const top = src[i00] + (src[i10] - src[i00]) * fx;
        const bot = src[i01] + (src[i11] - src[i01]) * fx;
        return top + (bot - top) * fy;
      }

      // Continuous per-pixel optical glass edge morphing (convex meniscus bevel + organic glass refraction)
      const innerFlat = 0.42;
      const invBevel = 1.0 / (1.0 - innerFlat);
      const haloA = pointer.haloAlpha;

      for (let py = 0; py < patchSize; py++) {
        const dy = py - halfSpan + 0.5;
        for (let px = 0; px < patchSize; px++) {
          const dx = px - halfSpan + 0.5;
          const dist = Math.hypot(dx, dy);

          if (dist > cr + 0.8) continue;

          const u = dist / cr;
          let srcX = cx + dx;
          let srcY = cy + dy;

          if (u > innerFlat) {
            const theta = Math.atan2(dy, dx);
            const s = clamp((u - innerFlat) * invBevel, 0, 1);
            const s2 = s * s;
            const s3 = s2 * s;

            // Smooth C2 convex glass meniscus magnification & edge bending
            const glassRipple =
              1.0 +
              (Math.sin(theta * 4.0 + 0.4) * 0.10 +
                Math.cos(theta * 6.0 - 0.7) * 0.07) *
              s;
            const radialPull = (0.26 * s2 + 0.14 * s3) * glassRipple;
            const scaleR = 1.0 - radialPull;

            // Subtle tangential glass-thickness refraction twist along the curved glass rim
            const invDist = dist > 0.0001 ? 1.0 / dist : 0;
            const nx = dx * invDist;
            const ny = dy * invDist;
            const tx = -ny;
            const ty = nx;
            const tangShift =
              (Math.cos(theta * 3.0 + 0.6) * 0.036 -
                Math.sin(theta * 5.0 - 0.8) * 0.026) *
              s2 *
              cr;

            srcX = cx + dx * scaleR + tx * tangShift;
            srcY = cy + dy * scaleR + ty * tangShift;
            const chromaShift = s2 * 1.35 * dpr;

            const dIdx = (py * patchSize + px) * 4;
            dst[dIdx] = sampleBilinear(srcX - nx * chromaShift, srcY - ny * chromaShift, 0);
            dst[dIdx + 1] = sampleBilinear(srcX, srcY, 1);
            dst[dIdx + 2] = sampleBilinear(srcX + nx * chromaShift, srcY + ny * chromaShift, 2);
            const edgeCov = clamp(cr + 0.5 - dist, 0, 1);
            dst[dIdx + 3] = Math.round(255 * edgeCov * haloA);
          } else {
            const dIdx = (py * patchSize + px) * 4;
            dst[dIdx] = sampleBilinear(srcX, srcY, 0);
            dst[dIdx + 1] = sampleBilinear(srcX, srcY, 1);
            dst[dIdx + 2] = sampleBilinear(srcX, srcY, 2);
            dst[dIdx + 3] = Math.round(255 * haloA);
          }
        }
      }

      pCtx.putImageData(dstImg, 0, 0);
      ctx2d.drawImage(lensPatchCanvas, Math.round(cx - halfSpan), Math.round(cy - halfSpan));

      // Subtle optical glass meniscus lighting & refractive rim gradient
      ctx2d.save();
      ctx2d.globalAlpha = haloA;
      ctx2d.beginPath();
      ctx2d.arc(cx, cy, cr, 0, Math.PI * 2);
      ctx2d.clip();

      const glassGrad = ctx2d.createRadialGradient(
        cx - cr * 0.22,
        cy - cr * 0.22,
        cr * 0.1,
        cx,
        cy,
        cr
      );
      glassGrad.addColorStop(0, 'rgba(255, 255, 255, 0.07)');
      glassGrad.addColorStop(0.64, 'rgba(255, 255, 255, 0.0)');
      glassGrad.addColorStop(0.84, 'rgba(235, 245, 255, 0.16)');
      glassGrad.addColorStop(0.94, 'rgba(175, 192, 208, 0.28)');
      glassGrad.addColorStop(1.0, 'rgba(255, 255, 255, 0.50)');
      ctx2d.fillStyle = glassGrad;
      ctx2d.fillRect(cx - cr, cy - cr, cr * 2, cr * 2);
      ctx2d.restore();

      // Crisp, delicate glass droplet edge & curved specular highlights
      ctx2d.save();
      ctx2d.scale(dpr, dpr);

      // Soft outer refractive shadow drop
      ctx2d.strokeStyle = `rgba(40, 35, 28, ${0.18 * haloA})`;
      ctx2d.lineWidth = 2.2;
      ctx2d.beginPath();
      ctx2d.arc(lx, ly + 1.2, r + 0.4, 0, Math.PI * 2);
      ctx2d.stroke();

      // Bright glass rim
      ctx2d.strokeStyle = `rgba(255, 255, 255, ${0.58 * haloA})`;
      ctx2d.lineWidth = 1.2;
      ctx2d.beginPath();
      ctx2d.arc(lx, ly, r, 0, Math.PI * 2);
      ctx2d.stroke();

      ctx2d.restore();
    }

    // PASS 2: Restore Normal exterior meshes (banana) and render onto webglCanvas
    for (const m of normalMeshes) m.visible = true;
    xrayGroup.visible = false;
    camera.zoom = 1.0;
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);
  }

  // ============================================================================
  // GALLERY NAVIGATION, RESIZE, & MAIN LOOP
  // ============================================================================
  function updateGalleryUI() {
    const curr = EXHIBITS[currentExhibitIndex];
    const prevIdx = (currentExhibitIndex + EXHIBITS.length - 1) % EXHIBITS.length;
    const nextIdx = (currentExhibitIndex + 1) % EXHIBITS.length;

    const prev = EXHIBITS[prevIdx];
    const next = EXHIBITS[nextIdx];

    if (plaqueTitle) plaqueTitle.textContent = curr.title;
    if (plaqueSubtitle) plaqueSubtitle.textContent = curr.subtitle;

    // Populate side thumbnails: Mimosa retains its drawn SVG; 3D exhibits use static miniature 3D canvases
    if (prev.id === 'mimosa') {
      prevIconEl.innerHTML = prev.iconSvg;
    } else {
      prevIconEl.innerHTML = '';
      prevIconEl.appendChild(prevThumbCanvas);
    }
    prevLabelEl.textContent = prev.shortLabel;

    if (next.id === 'mimosa') {
      nextIconEl.innerHTML = next.iconSvg;
    } else {
      nextIconEl.innerHTML = '';
      nextIconEl.appendChild(nextThumbCanvas);
    }
    nextLabelEl.textContent = next.shortLabel;

    renderSideThumbnails();

    // Show/hide WebGL canvas depending on whether current exhibit is 2D-only (Mimosa)
    webglCanvas.style.visibility = curr.id === 'mimosa' ? 'hidden' : 'visible';
    canvas2d.style.filter = 'none';
    // Hide system crosshair cursor on Cross Junction (3D person cursor) and Banana (magnifying glass lens)
    tileBezel.style.cursor = curr.id === 'junction' || curr.id === 'banana' ? 'none' : 'crosshair';
  }

  const tileFrameEl = document.getElementById('art-tile-frame');
  const plaqueEl = document.getElementById('art-plaque');
  let isTransitioning = false;

  function setNavButtonsDisabled(disabled) {
    navPrevBtn.disabled = disabled;
    navNextBtn.disabled = disabled;
    navPrevBtn.classList.toggle('is-disabled', disabled);
    navNextBtn.classList.toggle('is-disabled', disabled);
  }

  function switchExhibit(delta) {
    if (isTransitioning) return;
    isTransitioning = true;
    setNavButtonsDisabled(true);

    pointer.active = false;
    pointer.haloAlpha = 0;
    stopAllExhibitSounds();

    // 1. Current piece leaves by the left first
    if (tileFrameEl) {
      tileFrameEl.classList.remove('slide-prep-right', 'slide-in-center');
      tileFrameEl.classList.add('slide-out-left');
    }
    if (plaqueEl) {
      plaqueEl.classList.remove('slide-prep-right', 'slide-in-center');
      plaqueEl.classList.add('slide-out-left');
    }

    setTimeout(() => {
      // 2. Swap exhibit state while off-screen and position new piece on the right
      currentExhibitIndex = (currentExhibitIndex + delta + EXHIBITS.length) % EXHIBITS.length;
      if (presence && (!desktopRoom || (galleryRoom && galleryRoom.getMode() === 'exhibit'))) {
        presence.setExhibit(EXHIBITS[currentExhibitIndex].id, !desktopRoom);
      }
      resetMimosaEffects();
      ctx2d.clearRect(0, 0, canvas2d.width, canvas2d.height);
      updateGalleryUI();

      if (tileFrameEl) {
        tileFrameEl.classList.remove('slide-out-left');
        tileFrameEl.classList.add('slide-prep-right');
      }
      if (plaqueEl) {
        plaqueEl.classList.remove('slide-out-left');
        plaqueEl.classList.add('slide-prep-right');
      }

      // Force layout reflow so the right-side starting position is committed before sliding in
      if (tileFrameEl) void tileFrameEl.offsetWidth;

      // 3. New piece slides in from the right to the center
      requestAnimationFrame(() => {
        if (tileFrameEl) {
          tileFrameEl.classList.remove('slide-prep-right');
          tileFrameEl.classList.add('slide-in-center');
        }
        if (plaqueEl) {
          plaqueEl.classList.remove('slide-prep-right');
          plaqueEl.classList.add('slide-in-center');
        }

        setTimeout(() => {
          if (tileFrameEl) tileFrameEl.classList.remove('slide-in-center');
          if (plaqueEl) plaqueEl.classList.remove('slide-in-center');
          setNavButtonsDisabled(false);
          isTransitioning = false;
        }, 410);
      });
    }, 345);
  }

  navPrevBtn.addEventListener('click', () => switchExhibit(-1));
  navNextBtn.addEventListener('click', () => switchExhibit(1));

  window.addEventListener(
    'keydown',
    (e) => {
      if (
        (e.ctrlKey || e.metaKey) &&
        (e.key === '+' || e.key === '-' || e.key === '=' || e.key === '_' || e.key === '0')
      ) {
        e.preventDefault();
        return;
      }
      if (
        desktopRoom &&
        (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'ArrowUp' || e.key === 'ArrowDown')
      ) {
        return;
      }
      if (e.key === 'ArrowLeft') switchExhibit(-1);
      else if (e.key === 'ArrowRight') switchExhibit(1);
    },
    { passive: false }
  );

  // Prevent trackpad pinch-to-zoom, Ctrl/Cmd + scroll wheel zoom, and page wheel scrolling
  window.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
    },
    { passive: false }
  );

  // Prevent Safari / iOS / macOS WebKit trackpad & pinch gesture zoom
  ['gesturestart', 'gesturechange', 'gestureend'].forEach((evtName) => {
    window.addEventListener(
      evtName,
      (e) => {
        e.preventDefault();
      },
      { passive: false }
    );
  });

  // Prevent multi-touch pinch zoom & double-tap zoom across the entire document on mobile
  let lastTouchEndTime = 0;
  document.addEventListener(
    'touchstart',
    (e) => {
      if (e.touches.length > 1) {
        e.preventDefault();
      }
    },
    { passive: false }
  );

  document.addEventListener(
    'touchmove',
    (e) => {
      if (e.touches.length > 1) {
        e.preventDefault();
      }
    },
    { passive: false }
  );

  document.addEventListener(
    'touchend',
    (e) => {
      const now = performance.now();
      if (now - lastTouchEndTime <= 300) {
        e.preventDefault();
      }
      lastTouchEndTime = now;
    },
    { passive: false }
  );

  document.addEventListener(
    'dblclick',
    (e) => {
      e.preventDefault();
    },
    { passive: false }
  );

  function handleResize(force = false) {
    const rect = tileBezel.getBoundingClientRect();
    const newSize = Math.round(rect.width) || 460;
    if (!force && newSize === tileSize) return;

    tileSize = newSize;
    dpr = Math.min(window.devicePixelRatio || 1, 2);

    canvas2d.width = tileSize * dpr;
    canvas2d.height = tileSize * dpr;

    renderer.setPixelRatio(dpr);
    renderer.setSize(tileSize, tileSize, false);

    initMimosaTile();
  }

  function updatePointerFromEvent(clientX, clientY) {
    const rect = tileBezel.getBoundingClientRect();
    const lx = clientX - rect.left;
    const ly = clientY - rect.top;

    if (lx >= 0 && lx <= rect.width && ly >= 0 && ly <= rect.height) {
      pointer.prevX = pointer.x < -1000 ? lx : pointer.x;
      pointer.prevY = pointer.y < -1000 ? ly : pointer.y;
      pointer.vx = lx - pointer.prevX;
      pointer.vy = ly - pointer.prevY;
      pointer.x = lx;
      pointer.y = ly;
      pointer.ndcX = (lx / rect.width) * 2 - 1;
      pointer.ndcY = -(ly / rect.height) * 2 + 1;
      pointer.active = true;
    } else {
      pointer.active = false;
      pointer.haloAlpha = 0;
    }
  }

  const artFrame = document.getElementById('art-tile-frame');

  // Centers the top note in the space above the art frame.
  function placeTopNote() {
    if (!artFrame) return;
    const top = artFrame.getBoundingClientRect().top;
    document.documentElement.style.setProperty('--note-top', `${Math.max(0, top) / 2}px`);
  }

  window.addEventListener('resize', () => {
    handleResize(false);
    placeTopNote();
  });
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', () => {
      handleResize(false);
      placeTopNote();
    });
  }
  placeTopNote();
  window.addEventListener('load', placeTopNote);
  if (document.fonts) document.fonts.ready.then(placeTopNote);

  tileBezel.addEventListener('mousemove', (e) => {
    updatePointerFromEvent(e.clientX, e.clientY);
  });

  tileBezel.addEventListener('mouseleave', () => {
    pointer.active = false;
    pointer.haloAlpha = 0;
    pointer.x = -9999;
    pointer.y = -9999;
    pointer.vx = 0;
    pointer.vy = 0;
  });

  tileBezel.addEventListener(
    'touchstart',
    (e) => {
      if (e.touches.length === 1) {
        updatePointerFromEvent(e.touches[0].clientX, e.touches[0].clientY);
      }
    },
    { passive: true }
  );

  tileBezel.addEventListener(
    'touchmove',
    (e) => {
      e.preventDefault();
      if (e.touches.length === 1) {
        updatePointerFromEvent(e.touches[0].clientX, e.touches[0].clientY);
      }
    },
    { passive: false }
  );

  tileBezel.addEventListener('touchend', () => {
    pointer.active = false;
    pointer.haloAlpha = 0;
  });

  function animate(now) {
    const dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;
    const timeSec = now / 1000;
    const roomFrame = galleryRoom && galleryRoom.takesFrame();

    if (roomFrame && galleryRoom.blocksExhibit()) {
      galleryRoom.update(dt);
      requestAnimationFrame(animate);
      return;
    }

    if (pointer.active) {
      pointer.haloAlpha = lerp(pointer.haloAlpha, 1, clamp(dt * 14, 0, 1));
    } else {
      pointer.haloAlpha = 0;
    }

    const activeId = EXHIBITS[currentExhibitIndex].id;
    if (activeId === 'mimosa') {
      updateAndRenderMimosa(dt, timeSec);
    } else if (activeId === 'ducks') {
      updateAndRenderDucks(dt, timeSec);
    } else if (activeId === 'junction') {
      updateAndRenderJunction(dt);
    } else if (activeId === 'banana') {
      updateAndRenderBanana(dt, timeSec);
    }

    // Decay pointer velocity when stationary
    pointer.vx *= 0.8;
    pointer.vy *= 0.8;

    if (roomFrame) galleryRoom.update(dt);
    requestAnimationFrame(animate);
  }

  // Initialize all 4 exhibits
  handleResize(true);
  setupDuckScene();
  setupJunctionScene();
  setupBananaScene();
  updateGalleryUI();

  // Support URL hash for direct exhibit preview (#0, #1, #2, #3, and optional -hover)
  const rawHash = window.location.hash.replace('#', '');
  const hashIdx = parseInt(rawHash, 10);
  if (!isNaN(hashIdx) && hashIdx >= 0 && hashIdx < EXHIBITS.length) {
    currentExhibitIndex = hashIdx;
    updateGalleryUI();
    if (rawHash.includes('hover')) {
      const rect = tileBezel.getBoundingClientRect();
      // Place synthetic hover cursor right on the active subject (e.g. crosswalk lane or banana center)
      const hx = hashIdx === 2 ? rect.left + rect.width * 0.38 : rect.left + rect.width * 0.49;
      const hy = hashIdx === 2 ? rect.top + rect.height * 0.34 : rect.top + rect.height * 0.48;
      updatePointerFromEvent(hx, hy);
      bananaState.smoothX = pointer.x;
      bananaState.smoothY = pointer.y;
      pointer.haloAlpha = 1;
    }
  }

  presence = createPresence();
  const backBtn = document.getElementById('back-to-gallery');
  const roomCanvas = document.getElementById('room-canvas');

  function settleExhibit(id) {
    const idx = EXHIBITS.findIndex((exhibit) => exhibit.id === id);
    if (idx >= 0) currentExhibitIndex = idx;
    document.documentElement.classList.remove('in-room', 'boot-room');
    handleResize(true);
    placeTopNote();
    updateGalleryUI();
    if (presence) presence.setExhibit(id, false);
  }

  window.matchMedia('(min-width: 1024px) and (pointer: fine)').addEventListener('change', () => {
    location.reload();
  });

  if (desktopRoom && roomCanvas) {
    galleryRoom = createGalleryRoom({
      THREE,
      loader: gltfLoader,
      canvas: roomCanvas,
      onArrive: settleExhibit,
      onSettled: () => {
        if (backBtn) backBtn.hidden = false;
      }
    });

    if (backBtn) {
      backBtn.addEventListener('click', () => {
        if (!galleryRoom || galleryRoom.getMode() === 'fly' || galleryRoom.getMode() === 'room') return;
        backBtn.hidden = true;
        const exhibitId = EXHIBITS[currentExhibitIndex].id;
        if (presence) presence.setExhibit(null);
        document.documentElement.classList.add('in-room');
        document.documentElement.classList.remove('boot-room');
        const snap = presence.snapshot();
        galleryRoom.flyHome(exhibitId);
        snap.then((data) => galleryRoom.setCrowd(data));
      });
    }

    const hashOpensExhibit = /^#\d/.test(window.location.hash);
    if (hashOpensExhibit) {
      if (backBtn) backBtn.hidden = false;
      if (presence) presence.setExhibit(EXHIBITS[currentExhibitIndex].id, false);
    } else {
      document.documentElement.classList.add('in-room');
      document.documentElement.classList.remove('boot-room');
      galleryRoom.start();
      presence.snapshot().then((data) => galleryRoom.setCrowd(data));
    }
  } else if (presence) {
    presence.setExhibit(EXHIBITS[currentExhibitIndex].id, true);
  }

  requestAnimationFrame(animate);
})();


