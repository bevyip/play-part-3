/**
 * Overhead gallery. One WebGL renderer, created only on desktop.
 * The camera stays fixed until a floor zone in front of a piece starts a curve.
 * Other visitors are placed once per visit to this view and do not move.
 */

const HALF_X = 7.2;
const HALF_Z = 4.55;
const SPEED = 3.5;
const FLIGHT_SEC = 1.15;
const CHAR_HEIGHT = 1.22;
const PHONE_LENGTH = 0.22;
const MAX_PER_WALL = 8;
// The overview frames the room at this height; the walls themselves run up to
// a ceiling above the camera so no view ever leaves the room.
const WALL_HEIGHT = 3.6;
const ROOM_TOP = 60;
const HANDOFF = 0.75;
const WALL_THICK = 0.2;
const ART_SIZE = 1.5;
const ART_Y = 1.65;
const ART_INNER = 0.94;
const PLANK_TILE_M = 3.2;
const WALL_TILE_M = 2.4;
const SHADOW_SPAN = 1.3;
const CORNER_SHADE_W = 0.9;
const WALK_RATE = 9;

// Bends the unrigged character in its own mesh space (about 1.9 units tall,
// centered on the origin): legs swing from the hip, arms from the shoulder.
const WALK_GLSL = `
uniform float uSwing;
float walkAngle(vec3 p, out float pivot) {
  float leg = 1.0 - smoothstep(-0.42, -0.24, p.y);
  float arm = smoothstep(0.26, 0.32, abs(p.x)) * (1.0 - smoothstep(0.32, 0.42, p.y)) * (1.0 - leg);
  if (arm > 0.001) {
    pivot = 0.34;
    return -uSwing * sign(p.x) * arm * 0.6;
  }
  pivot = -0.3;
  float side = smoothstep(-0.04, 0.04, p.x) * 2.0 - 1.0;
  return uSwing * side * leg * 0.5;
}
vec3 walkRotate(vec3 v, float a, float pivot) {
  float c = cos(a);
  float s = sin(a);
  float y = v.y - pivot;
  return vec3(v.x, y * c - v.z * s + pivot, y * s + v.z * c);
}
`;

// Static pose for visitors holding a phone in their +x hand: forearm raised
// from the elbow, whole arm tipped slightly forward from the shoulder.
const HOLD_GLSL = `
vec3 holdRotate(vec3 v, float a, float pivot) {
  float c = cos(a);
  float s = sin(a);
  float y = v.y - pivot;
  return vec3(v.x, y * c - v.z * s + pivot, y * s + v.z * c);
}
vec3 holdPose(vec3 v, vec3 p, bool isNormal) {
  float arm = smoothstep(0.26, 0.32, p.x) * (1.0 - smoothstep(0.32, 0.42, p.y)) * smoothstep(-0.46, -0.38, p.y);
  if (arm < 0.001) return v;
  float fore = 1.0 - smoothstep(-0.04, 0.08, p.y);
  vec3 r = holdRotate(v, -1.3 * fore * arm, isNormal ? 0.0 : 0.02);
  r = holdRotate(r, -0.3 * arm, isNormal ? 0.0 : 0.34);
  if (!isNormal) r.x -= 0.1 * fore * arm;
  return r;
}
`;

const WALLS = {
  mimosa: { id: "mimosa", px: 0, pz: -1 },
  ducks: { id: "ducks", px: 0, pz: 1 },
  banana: { id: "banana", px: -1, pz: 0 },
  junction: { id: "junction", px: 1, pz: 0 },
};

const OVERVIEW_FOV = 41;
// A tiny z offset keeps lookAt stable when looking straight down.
const OVERVIEW_POS = { x: 0, y: 15.8, z: 0.01 };
const OVERVIEW_TARGET = { x: 0, y: 0, z: 0 };

const ARROWS = new Set(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]);

function smoothstep(t) {
  const x = t < 0 ? 0 : t > 1 ? 1 : t;
  return x * x * (3 - 2 * x);
}

function woodPlanks() {
  const size = 1024;
  const rows = 14;
  const rowH = size / rows;
  const tones = ["#d8b88d", "#d3b085", "#dcbe94", "#cfab7e", "#d6b48a"];
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  let seed = 7;
  const rand = () => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };

  for (let r = 0; r < rows; r += 1) {
    const y = r * rowH;
    let x = -rand() * 700;
    while (x < size) {
      const len = 560 + rand() * 520;
      ctx.fillStyle = tones[Math.floor(rand() * tones.length)];
      ctx.fillRect(x, y, len, rowH);

      for (let g = 0; g < 18; g += 1) {
        const gy = y + 4 + rand() * (rowH - 8);
        ctx.strokeStyle = `rgba(120, 82, 46, ${0.04 + rand() * 0.07})`;
        ctx.lineWidth = 0.6 + rand() * 1.6;
        ctx.beginPath();
        ctx.moveTo(x, gy);
        ctx.bezierCurveTo(
          x + len * 0.3,
          gy + (rand() - 0.5) * 6,
          x + len * 0.7,
          gy + (rand() - 0.5) * 6,
          x + len,
          gy,
        );
        ctx.stroke();
      }

      ctx.fillStyle = "rgba(92, 62, 36, 0.22)";
      ctx.fillRect(x, y, 2, rowH);
      x += len;
    }
    ctx.fillStyle = "rgba(92, 62, 36, 0.26)";
    ctx.fillRect(0, y, size, 2);
  }
  return canvas;
}

function contactShadow() {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const art = size / SHADOW_SPAN;
  const left = (size - art) / 2;
  ctx.shadowColor = "rgba(38, 30, 22, 0.55)";
  ctx.shadowBlur = 14;
  ctx.shadowOffsetX = 4000;
  ctx.shadowOffsetY = 9;
  ctx.fillStyle = "#000";
  ctx.fillRect(left - 4000, left, art, art);
  return canvas;
}

function cornerShade() {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 4;
  const ctx = canvas.getContext("2d");
  const fade = ctx.createLinearGradient(0, 0, 128, 0);
  fade.addColorStop(0, "rgba(48, 38, 28, 0.34)");
  fade.addColorStop(0.25, "rgba(48, 38, 28, 0.14)");
  fade.addColorStop(1, "rgba(48, 38, 28, 0)");
  ctx.fillStyle = fade;
  ctx.fillRect(0, 0, 128, 4);
  return canvas;
}

function pointOnWall(wall, along, inset) {
  if (wall.pz !== 0) return { x: along, z: wall.pz * (HALF_Z - inset) };
  return { x: wall.px * (HALF_X - inset), z: along };
}

export function createGalleryRoom({
  THREE,
  loader,
  canvas,
  onArrive,
  onSettled,
}) {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#f2f1ed");

  const camera = new THREE.PerspectiveCamera(
    OVERVIEW_FOV,
    window.innerWidth / window.innerHeight,
    0.1,
    140,
  );
  const overviewPos = new THREE.Vector3(
    OVERVIEW_POS.x,
    OVERVIEW_POS.y,
    OVERVIEW_POS.z,
  );
  const overviewTarget = new THREE.Vector3(
    OVERVIEW_TARGET.x,
    OVERVIEW_TARGET.y,
    OVERVIEW_TARGET.z,
  );

  // Lowest height at which the whole inside of the wall tops fits the window.
  function fitOverview() {
    const tanHalf = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    const dist = Math.max(HALF_Z / tanHalf, HALF_X / (tanHalf * camera.aspect));
    overviewPos.y = Math.min(WALL_HEIGHT + dist, ROOM_TOP - 0.5);
  }
  fitOverview();
  camera.position.copy(overviewPos);
  camera.lookAt(overviewTarget);

  scene.add(new THREE.HemisphereLight("#ffffff", "#9c968c", 1.9));
  const overhead = new THREE.DirectionalLight("#ffffff", 1.1);
  overhead.position.set(2, 10, 3);
  scene.add(overhead);

  const floorMap = new THREE.CanvasTexture(woodPlanks());
  floorMap.colorSpace = THREE.SRGBColorSpace;
  floorMap.wrapS = THREE.RepeatWrapping;
  floorMap.wrapT = THREE.RepeatWrapping;
  floorMap.anisotropy = 4;
  floorMap.repeat.set((HALF_X * 2) / PLANK_TILE_M, (HALF_Z * 2) / PLANK_TILE_M);
  const floor = new THREE.Mesh(
    new THREE.BoxGeometry(HALF_X * 2, 0.08, HALF_Z * 2),
    new THREE.MeshBasicMaterial({ map: floorMap }),
  );
  floor.position.y = -0.06;
  scene.add(floor);

  const textureLoader = new THREE.TextureLoader();
  const plasterCopies = [];
  const plaster = textureLoader.load("media/white-wall.jpg", () => {
    for (const tex of plasterCopies) tex.needsUpdate = true;
  });
  plaster.colorSpace = THREE.SRGBColorSpace;
  plaster.wrapS = THREE.RepeatWrapping;
  plaster.wrapT = THREE.RepeatWrapping;
  plaster.anisotropy = 4;
  function plasterFor(width) {
    const tex = plaster.clone();
    tex.repeat.set(width / WALL_TILE_M, ROOM_TOP / WALL_TILE_M);
    plasterCopies.push(tex);
    return new THREE.MeshBasicMaterial({
      map: tex,
      color: new THREE.Color(1.07, 1.07, 1.07),
    });
  }
  const longWallMat = plasterFor(HALF_X * 2);
  const shortWallMat = plasterFor(HALF_Z * 2);
  const longWall = new THREE.BoxGeometry(
    HALF_X * 2 + WALL_THICK * 2,
    ROOM_TOP,
    WALL_THICK,
  );
  const shortWall = new THREE.BoxGeometry(WALL_THICK, ROOM_TOP, HALF_Z * 2);
  const ceiling = new THREE.Mesh(
    new THREE.PlaneGeometry(HALF_X * 2, HALF_Z * 2),
    new THREE.MeshBasicMaterial({ color: "#f4f3ef" }),
  );
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.y = ROOM_TOP;
  scene.add(ceiling);

  const shadowMat = new THREE.MeshBasicMaterial({
    map: new THREE.CanvasTexture(contactShadow()),
    transparent: true,
    depthWrite: false,
  });
  const shadowGeo = new THREE.PlaneGeometry(
    ART_SIZE * SHADOW_SPAN,
    ART_SIZE * SHADOW_SPAN,
  );

  const cornerTexLeft = new THREE.CanvasTexture(cornerShade());
  const cornerTexRight = cornerTexLeft.clone();
  cornerTexRight.wrapS = THREE.RepeatWrapping;
  cornerTexRight.repeat.x = -1;
  cornerTexRight.offset.x = 1;
  cornerTexRight.needsUpdate = true;
  const cornerMatLeft = new THREE.MeshBasicMaterial({
    map: cornerTexLeft,
    transparent: true,
    depthWrite: false,
  });
  const cornerMatRight = new THREE.MeshBasicMaterial({
    map: cornerTexRight,
    transparent: true,
    depthWrite: false,
  });
  const cornerGeo = new THREE.PlaneGeometry(CORNER_SHADE_W, ROOM_TOP);
  const frameSide = new THREE.MeshLambertMaterial({ color: "#cfc8bc" });
  const frameTop = new THREE.MeshBasicMaterial({ color: "#f7f5f0" });
  const artFaceGeo = new THREE.PlaneGeometry(
    ART_SIZE * ART_INNER,
    ART_SIZE * ART_INNER,
  );

  for (const id of Object.keys(WALLS)) {
    const wall = WALLS[id];

    const slab =
      wall.pz !== 0
        ? new THREE.Mesh(longWall, longWallMat)
        : new THREE.Mesh(shortWall, shortWallMat);
    const back = pointOnWall(wall, 0, -WALL_THICK / 2);
    slab.position.set(back.x, ROOM_TOP / 2, back.z);
    scene.add(slab);

    const map = textureLoader.load(`media/room/${id}.png`);
    map.colorSpace = THREE.SRGBColorSpace;
    map.anisotropy = 4;
    map.repeat.set(0.96, 0.96);
    map.offset.set(0.02, 0.02);

    const face = new THREE.MeshLambertMaterial({ map });
    const art = new THREE.Mesh(
      new THREE.BoxGeometry(ART_SIZE, ART_SIZE, 0.08),
      [frameSide, frameSide, frameTop, frameSide, frameTop, frameSide],
    );
    const picture = new THREE.Mesh(artFaceGeo, face);
    picture.position.z = 0.041;
    art.add(picture);
    const hang = pointOnWall(wall, 0, 0.1);
    art.position.set(hang.x, ART_Y, hang.z);
    art.lookAt(0, ART_Y, 0);
    scene.add(art);

    const drop = new THREE.Mesh(shadowGeo, shadowMat);
    const behind = pointOnWall(wall, 0, 0.004);
    drop.position.set(behind.x, ART_Y, behind.z);
    drop.lookAt(0, ART_Y, 0);
    scene.add(drop);

    const halfLen = wall.pz !== 0 ? HALF_X : HALF_Z;
    for (const end of [-1, 1]) {
      const p = pointOnWall(wall, end * (halfLen - CORNER_SHADE_W / 2), 0.003);
      const shade = new THREE.Mesh(cornerGeo, cornerMatLeft);
      shade.position.set(p.x, ROOM_TOP / 2, p.z);
      shade.lookAt(p.x - wall.px, ROOM_TOP / 2, p.z - wall.pz);
      const localX = new THREE.Vector3(1, 0, 0).applyQuaternion(
        shade.quaternion,
      );
      const towardCorner = wall.pz !== 0 ? localX.x * end : localX.z * end;
      if (towardCorner > 0) shade.material = cornerMatRight;
      scene.add(shade);
    }
  }

  const whiteMat = new THREE.MeshLambertMaterial({ color: "#f3f3f1" });
  const visitors = new THREE.Group();
  scene.add(visitors);

  const walkSwing = { value: 0 };
  let walkPhase = 0;
  let walkAmount = 0;
  const playerMat = new THREE.MeshLambertMaterial({ color: "#f3f3f1" });
  const visitorMat = new THREE.MeshLambertMaterial({ color: "#d2d2cf" });
  const holdingMat = visitorMat.clone();
  holdingMat.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", `#include <common>\n${HOLD_GLSL}`)
      .replace(
        "#include <beginnormal_vertex>",
        "#include <beginnormal_vertex>\nobjectNormal = holdPose(objectNormal, position, true);",
      )
      .replace(
        "#include <begin_vertex>",
        "#include <begin_vertex>\ntransformed = holdPose(transformed, position, false);",
      );
  };
  playerMat.onBeforeCompile = (shader) => {
    shader.uniforms.uSwing = walkSwing;
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", `#include <common>\n${WALK_GLSL}`)
      .replace(
        "#include <beginnormal_vertex>",
        `#include <beginnormal_vertex>
        float walkPivot;
        float walkA = walkAngle(position, walkPivot);
        objectNormal = walkRotate(objectNormal, walkA, 0.0);`,
      )
      .replace(
        "#include <begin_vertex>",
        `#include <begin_vertex>
        transformed = walkRotate(transformed, walkA, walkPivot);`,
      );
  };

  const held = new Set();
  let mode = "exhibit";
  let characterTemplate = null;
  let phoneTemplate = null;
  let phoneDone = false;
  let templatesReady = false;
  let player = null;
  let pendingCrowd = null;
  let flight = null;

  const fromPos = new THREE.Vector3();
  const toPos = new THREE.Vector3();
  const fromAim = { yaw: 0, pitch: 0 };
  const toAim = { yaw: 0, pitch: 0 };
  const pathAim = { yaw: 0, pitch: 0 };
  const artTarget = new THREE.Vector3();

  function aimFor(pos, target, out) {
    const dx = target.x - pos.x;
    const dy = target.y - pos.y;
    const dz = target.z - pos.z;
    out.yaw = Math.atan2(-dx, -dz);
    out.pitch = Math.atan2(dy, Math.hypot(dx, dz));
    return out;
  }

  function applyAim(yaw, pitch) {
    camera.rotation.set(pitch, yaw, 0, "YXZ");
  }

  function resize() {
    camera.aspect = window.innerWidth / Math.max(window.innerHeight, 1);
    camera.updateProjectionMatrix();
    fitOverview();
    if (mode === "room") lookOverview();
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.setSize(window.innerWidth, window.innerHeight, false);
  }

  function showCanvasSharp() {
    canvas.classList.remove("is-fading");
    canvas.classList.add("is-on");
  }

  function closePose(wall) {
    const eye = pointOnWall(wall, 0, 3.05);
    const look = pointOnWall(wall, 0, 0.02);
    return {
      pos: new THREE.Vector3(eye.x, ART_Y, eye.z),
      target: new THREE.Vector3(look.x, ART_Y, look.z),
    };
  }

  function lookOverview() {
    camera.position.copy(overviewPos);
    camera.lookAt(overviewTarget);
  }

  function fitUpright(object, height) {
    const box = new THREE.Box3().setFromObject(object);
    const size = box.getSize(new THREE.Vector3());
    const scale = height / Math.max(size.y, 0.001);
    object.scale.multiplyScalar(scale);
    object.updateMatrixWorld(true);
    const fitted = new THREE.Box3().setFromObject(object);
    object.position.set(
      object.position.x - (fitted.min.x + fitted.max.x) / 2,
      object.position.y - fitted.min.y,
      object.position.z - (fitted.min.z + fitted.max.z) / 2,
    );
    return object;
  }

  function buildPerson(withPhone) {
    const root = new THREE.Group();
    root.add(characterTemplate.clone(true));
    if (withPhone && phoneTemplate) {
      const phone = phoneTemplate.clone(true);
      phone.position.set(0.19, 0.6, 0.22);
      phone.rotation.set(-0.5, 0.3, 0, "YXZ");
      root.add(phone);
    }
    return root;
  }

  function spawnPlayer() {
    player = buildPerson(false);
    player.traverse((obj) => {
      if (obj.isMesh) obj.material = playerMat;
    });
    player.position.set(0, 0, 0);
    scene.add(player);
  }

  function applyCrowd() {
    while (visitors.children.length) visitors.remove(visitors.children[0]);
    const data = pendingCrowd || {};
    for (const id of Object.keys(WALLS)) {
      const wall = WALLS[id];
      const flags = Array.isArray(data[id])
        ? data[id].slice(0, MAX_PER_WALL)
        : [];
      const rowCount = Math.min(flags.length, 4);
      flags.forEach((isMobile, index) => {
        const col = index % 4;
        const row = Math.floor(index / 4);
        const countInRow = row === 0 ? rowCount : flags.length - 4;
        const along = (col - (countInRow - 1) / 2) * 0.76;
        const spot = pointOnWall(wall, along, 1.45 + row * 0.68);
        const person = buildPerson(!!isMobile);
        person.children[0].traverse((obj) => {
          if (obj.isMesh) obj.material = isMobile ? holdingMat : visitorMat;
        });
        person.position.set(spot.x, 0, spot.z);
        person.rotation.y = Math.atan2(wall.px, wall.pz);
        person.updateMatrixWorld(true);
        person.traverse((obj) => {
          obj.matrixAutoUpdate = false;
          obj.updateMatrix();
        });
        visitors.add(person);
      });
    }
  }

  // Draws a front view of each visitor type once into the legend <img>s, so
  // the legend costs nothing per frame.
  function renderLegend() {
    const slots = [
      ["legend-desktop", false],
      ["legend-mobile", true],
    ];
    const w = 240;
    const h = 330;
    const target = new THREE.WebGLRenderTarget(w, h);
    target.texture.colorSpace = THREE.SRGBColorSpace;
    const stage = new THREE.Scene();
    stage.add(new THREE.HemisphereLight("#ffffff", "#9c968c", 1.9));
    const key = new THREE.DirectionalLight("#ffffff", 1.1);
    key.position.set(1.5, 3, 4);
    stage.add(key);
    const cam = new THREE.PerspectiveCamera(24, w / h, 0.1, 20);
    cam.position.set(0, 1.05, 3.6);
    cam.lookAt(0, 0.62, 0);

    const pixels = new Uint8Array(w * h * 4);
    const out = document.createElement("canvas");
    out.width = w;
    out.height = h;
    const ctx = out.getContext("2d");
    const image = ctx.createImageData(w, h);
    const clearColor = renderer.getClearColor(new THREE.Color());
    const clearAlpha = renderer.getClearAlpha();
    renderer.setClearColor(0x000000, 0);

    for (const [id, withPhone] of slots) {
      const el = document.getElementById(id);
      if (!el) continue;
      const person = buildPerson(withPhone);
      person.children[0].traverse((obj) => {
        if (obj.isMesh) obj.material = withPhone ? holdingMat : visitorMat;
      });
      if (person.children[1]) person.children[1].rotation.x = -1.15;
      stage.add(person);
      renderer.setRenderTarget(target);
      renderer.render(stage, cam);
      renderer.readRenderTargetPixels(target, 0, 0, w, h, pixels);
      const row = w * 4;
      for (let y = 0; y < h; y++) {
        image.data.set(
          pixels.subarray((h - 1 - y) * row, (h - y) * row),
          y * row,
        );
      }
      ctx.putImageData(image, 0, 0);
      el.src = out.toDataURL("image/png");
      stage.remove(person);
    }

    renderer.setRenderTarget(null);
    renderer.setClearColor(clearColor, clearAlpha);
    target.dispose();
  }

  function markReady() {
    if (!characterTemplate || !phoneDone || templatesReady) return;
    templatesReady = true;
    if (!player) spawnPlayer();
    applyCrowd();
    renderLegend();
  }

  loader.load("media/character.glb", (gltf) => {
    const model = gltf.scene;
    model.traverse((obj) => {
      if (obj.isMesh) {
        obj.material = whiteMat;
        obj.castShadow = false;
        obj.receiveShadow = false;
      }
    });
    characterTemplate = fitUpright(model, CHAR_HEIGHT);
    markReady();
  });

  loader.load(
    "media/phone.glb",
    (gltf) => {
      const model = gltf.scene;
      model.traverse((obj) => {
        if (obj.isMesh) {
          obj.castShadow = false;
          obj.receiveShadow = false;
        }
      });
      // The model ships tilted 45° toward the viewer; lay it flat with the screen facing up.
      model.rotation.x = -Math.PI / 4;
      model.updateMatrixWorld(true);
      const size = new THREE.Box3()
        .setFromObject(model)
        .getSize(new THREE.Vector3());
      const long = Math.max(size.x, size.y, size.z);
      model.scale.multiplyScalar(PHONE_LENGTH / Math.max(long, 0.001));
      model.updateMatrixWorld(true);
      const fitted = new THREE.Box3().setFromObject(model);
      model.position.sub(fitted.getCenter(new THREE.Vector3()));
      const wrap = new THREE.Group();
      wrap.add(model);
      phoneTemplate = wrap;
      phoneDone = true;
      markReady();
    },
    undefined,
    () => {
      phoneDone = true;
      markReady();
    },
  );

  function triggersHit(x, z) {
    for (const id of Object.keys(WALLS)) {
      const wall = WALLS[id];
      const near = pointOnWall(wall, 0, 0.5);
      const far = pointOnWall(wall, 0, 2.15);
      if (wall.pz !== 0) {
        const minZ = Math.min(near.z, far.z);
        const maxZ = Math.max(near.z, far.z);
        if (x >= -1.25 && x <= 1.25 && z >= minZ && z <= maxZ) return wall;
      } else {
        const minX = Math.min(near.x, far.x);
        const maxX = Math.max(near.x, far.x);
        if (z >= -1.25 && z <= 1.25 && x >= minX && x <= maxX) return wall;
      }
    }
    return null;
  }

  // Paths are described from the overview (n = 0) to the art (n = 1); the
  // return flight runs the same path backwards. The turn leads the tilt, so
  // the room spins while still seen from above instead of panning bare walls.
  function beginFlight(wall, towardHome) {
    const pose = closePose(wall);
    fromPos.copy(overviewPos);
    toPos.copy(pose.pos);
    aimFor(overviewPos, overviewTarget, fromAim);
    artTarget.copy(pose.target);
    aimFor(pose.pos, pose.target, toAim);
    let turn = toAim.yaw - fromAim.yaw;
    while (turn > Math.PI) turn -= Math.PI * 2;
    while (turn < -Math.PI) turn += Math.PI * 2;
    toAim.yaw = fromAim.yaw + turn;
    releaseKeys();
    flight = { t: 0, towardHome, wall, handed: false };
    mode = "fly";
    if (player) player.visible = false;
    canvas.style.opacity = "";
    showCanvasSharp();
    placeOnPath(towardHome ? 1 : 0);
  }

  function placeOnPath(n) {
    camera.position.lerpVectors(fromPos, toPos, n);
    const turned = 1 - (1 - n) * (1 - n);
    const towardArt = aimFor(camera.position, artTarget, pathAim).pitch;
    applyAim(
      fromAim.yaw + (toAim.yaw - fromAim.yaw) * turned,
      fromAim.pitch + (towardArt - fromAim.pitch) * turned,
    );
  }

  function finishFlight() {
    const arrived = flight;
    flight = null;
    if (arrived.towardHome) {
      mode = "room";
      lookOverview();
      if (player) {
        player.position.set(0, 0, 0);
        player.rotation.y = 0;
        player.visible = true;
      }
      return;
    }
    mode = "exhibit";
    canvas.classList.remove("is-on", "is-fading");
    canvas.style.opacity = "";
    onSettled();
  }

  // The exhibit is shown underneath partway in and the room fades off it
  // while still moving, so the miniature never fills the screen.
  function stepFlight(dt) {
    flight.t = Math.min(flight.t + dt / FLIGHT_SEC, 1);
    const eased = smoothstep(flight.t);
    placeOnPath(flight.towardHome ? 1 - eased : eased);
    if (!flight.towardHome) {
      if (!flight.handed && flight.t >= HANDOFF) {
        flight.handed = true;
        onArrive(flight.wall.id);
      }
      if (flight.handed) {
        canvas.style.opacity = String(
          Math.max(0, 1 - (flight.t - HANDOFF) / (1 - HANDOFF)),
        );
      }
    }
    if (flight.t >= 1) finishFlight();
  }

  function stepPlayer(dt) {
    let x = 0;
    let z = 0;
    if (held.has("ArrowLeft")) x -= 1;
    if (held.has("ArrowRight")) x += 1;
    if (held.has("ArrowUp")) z -= 1;
    if (held.has("ArrowDown")) z += 1;
    const moving = x !== 0 || z !== 0;
    walkAmount += ((moving ? 1 : 0) - walkAmount) * Math.min(1, dt * 10);
    if (moving) walkPhase += dt * WALK_RATE;
    walkSwing.value = Math.sin(walkPhase) * walkAmount;
    player.position.y = Math.abs(Math.sin(walkPhase)) * 0.035 * walkAmount;

    if (moving) {
      const len = Math.hypot(x, z) || 1;
      player.position.x = Math.min(
        HALF_X - 0.62,
        Math.max(-HALF_X + 0.62, player.position.x + (x / len) * SPEED * dt),
      );
      player.position.z = Math.min(
        HALF_Z - 0.62,
        Math.max(-HALF_Z + 0.62, player.position.z + (z / len) * SPEED * dt),
      );
      player.rotation.y = Math.atan2(x, z);
      const wall = triggersHit(player.position.x, player.position.z);
      if (wall) beginFlight(wall, false);
    }
  }

  function showKey(key, on) {
    const icon = document.querySelector(
      `.room-key-${key.slice(5).toLowerCase()}`,
    );
    if (icon) icon.classList.toggle("is-active", on);
  }

  function onKeyDown(event) {
    if (!ARROWS.has(event.key) || mode !== "room") return;
    event.preventDefault();
    held.add(event.key);
    showKey(event.key, true);
  }

  function onKeyUp(event) {
    if (!ARROWS.has(event.key)) return;
    held.delete(event.key);
    showKey(event.key, false);
  }

  function releaseKeys() {
    for (const key of held) showKey(key, false);
    held.clear();
  }

  window.addEventListener("blur", releaseKeys);

  for (const icon of document.querySelectorAll(".room-key")) {
    const dir = ["up", "down", "left", "right"].find((d) =>
      icon.classList.contains(`room-key-${d}`),
    );
    if (!dir) continue;
    const key = `Arrow${dir[0].toUpperCase()}${dir.slice(1)}`;
    icon.addEventListener("pointerdown", (event) => {
      if (mode !== "room") return;
      event.preventDefault();
      icon.setPointerCapture(event.pointerId);
      held.add(key);
      showKey(key, true);
    });
    const release = () => {
      held.delete(key);
      showKey(key, false);
    };
    icon.addEventListener("pointerup", release);
    icon.addEventListener("pointercancel", release);
    icon.addEventListener("lostpointercapture", release);
  }

  window.addEventListener("keydown", onKeyDown, true);
  window.addEventListener("keyup", onKeyUp, true);
  window.addEventListener("resize", resize);

  return {
    getMode() {
      return mode;
    },
    takesFrame() {
      return canvas.classList.contains("is-on");
    },
    blocksExhibit() {
      return mode === "room" || (mode === "fly" && !flight.handed);
    },
    start() {
      mode = "room";
      lookOverview();
      showCanvasSharp();
      if (player) {
        player.position.set(0, 0, 0);
        player.rotation.y = 0;
        player.visible = true;
      }
    },
    flyHome(exhibitId) {
      if (mode === "fly") return;
      const wall = WALLS[exhibitId] || WALLS.mimosa;
      beginFlight(wall, true);
    },
    setCrowd(data) {
      pendingCrowd = data || {};
      if (templatesReady) applyCrowd();
    },
    update(dt) {
      if (mode === "fly" && flight) stepFlight(dt);
      else if (mode === "room" && player) stepPlayer(dt);
      if (!document.hidden) renderer.render(scene, camera);
    },
  };
}
