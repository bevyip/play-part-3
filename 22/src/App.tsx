import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Hands, Results } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";
import { Palette, Loader2, X } from "lucide-react";

const COLORS = [
  { name: "rose", value: "#FFB3BA" },
  { name: "peach", value: "#FFDFBA" },
  { name: "butter", value: "#FFFFBA" },
  { name: "mint", value: "#BAFFC9" },
  { name: "sky", value: "#BAE1FF" },
  { name: "lavender", value: "#C9BAFF" },
  { name: "orchid", value: "#FFBAF0" },
  { name: "white", value: "#FFFFFF" },
];

/** Open-palm “finish drawing” hold duration (ms). */
const FINISH_PALM_HOLD_MS = 1000;

/**
 * Finger / thumb openness in MediaPipe normalized image space (0–1), not CSS px.
 * Fixed pixel deltas (e.g. 15px) required a larger relative spread on short viewports,
 * so open-palm rarely registered on phones.
 */
const RIGHT_FINGER_EXTEND_NORM = 0.015;
const RIGHT_THUMB_SPREAD_NORM = 0.018;

/** Polaroid 3D export backdrop: four dark, then four light. */
const POLAROID_BG_DARK_COUNT = 4;

const POLAROID_BG_SWATCHES: { value: string; name: string }[] = [
  { value: "#0e0e0e", name: "charcoal" },
  { value: "#12111a", name: "ink" },
  { value: "#0f1a17", name: "evergreen" },
  { value: "#1a1014", name: "plum" },
  { value: "#ede8e1", name: "parchment" },
  { value: "#dde5f5", name: "periwinkle" },
  { value: "#daeee0", name: "mint" },
  { value: "#f5dada", name: "blush" },
];

const DEFAULT_POLAROID_EXPORT_BG = POLAROID_BG_SWATCHES[0]!.value;

/** Pauses idle spin on main-scene meshes while an offscreen polaroid render runs. */
let polaroidSnapshotInProgress = false;

/**
 * One shared offscreen WebGL context for polaroid snapshots.
 * Creating + disposing a new WebGLRenderer on every slider/swatch/note update
 * exhausts the browser's context limit (especially mobile Safari); the GPU then
 * evicts the main canvas → white 3D view while the rest of the UI still paints.
 */
let polaroidExportRenderer: THREE.WebGLRenderer | null = null;

function getPolaroidExportRenderer(): THREE.WebGLRenderer {
  if (!polaroidExportRenderer) {
    polaroidExportRenderer = new THREE.WebGLRenderer({
      antialias: true,
      preserveDrawingBuffer: true,
      alpha: false,
    });
    polaroidExportRenderer.setSize(600, 600);
    polaroidExportRenderer.setPixelRatio(
      typeof window !== "undefined" ? Math.min(window.devicePixelRatio, 2) : 2,
    );
    polaroidExportRenderer.toneMapping = THREE.ACESFilmicToneMapping;
    polaroidExportRenderer.toneMappingExposure = 1.5;
    const el = polaroidExportRenderer.domElement;
    el.setAttribute("aria-hidden", "true");
    el.style.cssText =
      "position:fixed;left:0;top:0;width:1px;height:1px;opacity:0;pointer-events:none;";
    if (typeof document !== "undefined") {
      document.body.appendChild(el);
    }
  }
  return polaroidExportRenderer;
}

function disposePolaroidExportSceneObjects(scene: THREE.Scene) {
  while (scene.children.length > 0) {
    const o = scene.children[0];
    scene.remove(o);
    o.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.geometry?.dispose();
        const mat = child.material;
        if (Array.isArray(mat)) {
          mat.forEach((m) => m.dispose());
        } else {
          (mat as THREE.Material | undefined)?.dispose?.();
        }
      }
    });
  }
}

/** Renders only into a throwaway scene — never touches the live editor scene/canvas. */
function renderShapeToDataURL(
  mesh: THREE.Mesh,
  bgColor: string,
  angleYDeg: number,
  angleXDeg: number,
): string {
  polaroidSnapshotInProgress = true;
  const offScene = new THREE.Scene();
  try {
    offScene.background = new THREE.Color().setStyle(bgColor);

    const clone = mesh.clone(true);
    clone.castShadow = false;

    clone.rotation.set(0, 0, 0);
    clone.updateMatrixWorld(true);

    const box = new THREE.Box3().setFromObject(clone);
    const center = box.getCenter(new THREE.Vector3());
    clone.position.sub(center);
    clone.updateMatrixWorld(true);

    offScene.add(clone);

    const boxFinal = new THREE.Box3().setFromObject(clone);
    const size = boxFinal.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
    const offCam = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);

    const dist = maxDim * 2.4;
    const yRad = angleYDeg * (Math.PI / 180);
    const xRad = angleXDeg * (Math.PI / 180);
    const camX = dist * Math.cos(xRad) * Math.sin(yRad);
    const camY = dist * Math.sin(xRad);
    const camZ = dist * Math.cos(xRad) * Math.cos(yRad);
    offCam.position.set(camX, camY, camZ);
    offCam.lookAt(0, 0, 0);
    offCam.up.set(0, 1, 0);
    offCam.updateProjectionMatrix();

    offScene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const snapLight = new THREE.DirectionalLight(0xffffff, 2.2);
    snapLight.position.set(4, 8, 6);
    offScene.add(snapLight);
    const fillLight = new THREE.DirectionalLight(0xffe0f0, 0.7);
    fillLight.position.set(-4, 2, -3);
    offScene.add(fillLight);

    const offRenderer = getPolaroidExportRenderer();
    offRenderer.render(offScene, offCam);

    return offRenderer.domElement.toDataURL("image/png");
  } finally {
    disposePolaroidExportSceneObjects(offScene);
    polaroidSnapshotInProgress = false;
  }
}

const POLAROID_NOTE_FONT = "italic 22px Georgia, serif";
const POLAROID_NOTE_LINE_HEIGHT = 28;
const POLAROID_NOTE_MAX_WIDTH = 560 - 24 * 2;

function wrapPolaroidNoteLines(
  ctx: CanvasRenderingContext2D,
  text: string,
  maxWidth: number,
): string[] {
  ctx.font = POLAROID_NOTE_FONT;
  const display = text.trim() || "my little note";
  const words = display.split(/\s+/).filter(Boolean);
  const lines: string[] = [];
  let line = "";

  for (let n = 0; n < words.length; n++) {
    let word = words[n];
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width <= maxWidth) {
      line = test;
    } else {
      if (line) {
        lines.push(line);
        line = "";
      }
      while (word.length && ctx.measureText(word).width > maxWidth) {
        let i = 1;
        while (
          i <= word.length &&
          ctx.measureText(word.slice(0, i)).width <= maxWidth
        ) {
          i++;
        }
        i = Math.max(1, i - 1);
        lines.push(word.slice(0, i));
        word = word.slice(i);
      }
      if (word) line = word;
    }
  }
  if (line) lines.push(line);
  return lines.length ? lines : ["my little note"];
}

function clipPolaroidNoteToTwoLines(
  ctx: CanvasRenderingContext2D,
  lines: string[],
  maxWidth: number,
): string[] {
  if (lines.length <= 2) return lines;
  const first = lines[0];
  const ell = "…";
  let second = lines.slice(1).join(" ");
  while (second.length && ctx.measureText(second + ell).width > maxWidth) {
    second = second.slice(0, -1);
  }
  const trimmed = second.replace(/\s+$/, "");
  return [first, trimmed + ell];
}

/** Conservative cap so typed text cannot exceed two wrapped lines on the polaroid. */
function getPolaroidNoteMaxChars(): number {
  if (typeof document === "undefined") return 96;
  const ctx = document.createElement("canvas").getContext("2d");
  if (!ctx) return 96;
  ctx.font = POLAROID_NOTE_FONT;
  const wW = ctx.measureText("W").width || 12;
  const perLine = Math.max(8, Math.floor(POLAROID_NOTE_MAX_WIDTH / wW));
  return perLine * 2;
}

const POLAROID_NOTE_MAX_CHARS = getPolaroidNoteMaxChars();

const POLAROID_STAMP_MONTHS = [
  "JAN",
  "FEB",
  "MAR",
  "APR",
  "MAY",
  "JUN",
  "JUL",
  "AUG",
  "SEP",
  "OCT",
  "NOV",
  "DEC",
] as const;

function formatPolaroidStamp(d: Date): string {
  const mon = POLAROID_STAMP_MONTHS[d.getMonth()];
  const day = d.getDate();
  const year = d.getFullYear();
  let h = d.getHours();
  const m = d.getMinutes();
  const ampm = h >= 12 ? "PM" : "AM";
  h = h % 12;
  if (h === 0) h = 12;
  const mm = String(m).padStart(2, "0");
  return `made with love · ${mon} ${day} ${year}, ${h}:${mm} ${ampm}`;
}

/** Fine monochrome grain on the flattened polaroid (preview + saved PNG). */
function applyPolaroidFilmGrain(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  strength = 14,
) {
  const imageData = ctx.getImageData(0, 0, width, height);
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    const n = (Math.random() - 0.5) * strength;
    d[i] = Math.max(0, Math.min(255, d[i] + n));
    d[i + 1] = Math.max(0, Math.min(255, d[i + 1] + n));
    d[i + 2] = Math.max(0, Math.min(255, d[i + 2] + n));
  }
  ctx.putImageData(imageData, 0, 0);
}

function buildPolaroid(
  renderDataURL: string,
  noteText: string,
  _bgColor: string,
): Promise<string> {
  return new Promise((resolve) => {
    const W = 560;
    const H = 660;
    const pad = 24;
    const imgSize = W - pad * 2;

    const canvas = document.createElement("canvas");
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      resolve(renderDataURL);
      return;
    }

    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    // Rounded top corners only; bottom stays square (polaroid caption/footer block).
    ctx.roundRect(0, 0, W, H, [8, 8, 0, 0]);
    ctx.fill();

    ctx.fillStyle = "rgba(0,0,0,0.04)";
    ctx.fillRect(pad, pad, imgSize, 6);

    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, pad, pad, imgSize, imgSize);

      ctx.fillStyle = "#fafafa";
      ctx.fillRect(0, pad + imgSize, W, H - pad - imgSize);

      ctx.strokeStyle = "rgba(0,0,0,0.06)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad, pad + imgSize + 1);
      ctx.lineTo(W - pad, pad + imgSize + 1);
      ctx.stroke();

      const wrapped = wrapPolaroidNoteLines(
        ctx,
        noteText,
        POLAROID_NOTE_MAX_WIDTH,
      );
      const noteLines = clipPolaroidNoteToTwoLines(
        ctx,
        wrapped,
        POLAROID_NOTE_MAX_WIDTH,
      );

      ctx.font = POLAROID_NOTE_FONT;
      ctx.fillStyle = "#2a2a2a";
      ctx.textAlign = "center";

      const captionBandTop = pad + imgSize + 18;
      const captionBandBottom = H - 38;
      const midY = (captionBandTop + captionBandBottom) / 2;
      const nLines = noteLines.length;
      const noteTopBaseline =
        midY - ((nLines - 1) * POLAROID_NOTE_LINE_HEIGHT) / 2 + 4;

      noteLines.forEach((ln, i) => {
        ctx.fillText(
          ln,
          W / 2,
          noteTopBaseline + i * POLAROID_NOTE_LINE_HEIGHT,
        );
      });

      const stamp = formatPolaroidStamp(new Date());
      ctx.font = "11px monospace";
      ctx.fillStyle = "rgba(0,0,0,0.5)";
      ctx.textAlign = "center";
      ctx.fillText(stamp, W / 2, H - 20);

      applyPolaroidFilmGrain(ctx, W, H);

      resolve(canvas.toDataURL("image/png"));
    };
    img.src = renderDataURL;
  });
}

type GestureState = "IDLE" | "DRAW" | "FINISH";

type UILayoutTier = "desktop" | "tablet" | "mobile";

type PreviewLayout = {
  w: number;
  h: number;
  chrome: number;
  dot: number;
  tier: UILayoutTier;
};

/** Camera preview + related chrome; canvas buffer matches w×h (16:9). */
function computePreviewLayout(viewportWidth: number): PreviewLayout {
  if (viewportWidth < 640) {
    /** Compact, top-left on phone so it never intrudes on top-right UI. */
    const w = Math.floor(viewportWidth * 0.45);
    const h = Math.round((w * 9) / 16);
    return { w, h, chrome: 17, dot: 6, tier: "mobile" };
  }
  if (viewportWidth < 1024) {
    const w = 272;
    const h = Math.round((w * 9) / 16);
    return { w, h, chrome: 24, dot: 10, tier: "tablet" };
  }
  return { w: 320, h: 180, chrome: 28, dot: 11, tier: "desktop" };
}

export default function App() {
  const [activeColor, setActiveColor] = useState(COLORS[5].value);
  const [pinchedColor, setPinchedColor] = useState<string | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);

  const [previewLayout, setPreviewLayout] = useState<PreviewLayout>(() =>
    computePreviewLayout(
      typeof window !== "undefined" ? window.innerWidth : 1280,
    ),
  );
  const previewLayoutRef = useRef(previewLayout);
  previewLayoutRef.current = previewLayout;

  useEffect(() => {
    const sync = () => {
      const next = computePreviewLayout(window.innerWidth);
      previewLayoutRef.current = next;
      setPreviewLayout((prev) =>
        prev.w === next.w &&
        prev.h === next.h &&
        prev.chrome === next.chrome &&
        prev.dot === next.dot &&
        prev.tier === next.tier
          ? prev
          : next,
      );
    };
    sync();
    window.addEventListener("resize", sync);
    return () => window.removeEventListener("resize", sync);
  }, []);

  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvas2DRef = useRef<HTMLCanvasElement>(null);
  const canvas3DRef = useRef<HTMLCanvasElement>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);

  const activeColorRef = useRef(activeColor);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const meshesRef = useRef<THREE.Mesh[]>([]);
  const currentPointsRef = useRef<THREE.Vector2[]>([]);
  const smoothedCursorRef = useRef<THREE.Vector2 | null>(null);
  const isGrabActiveRef = useRef<boolean>(false);
  const grabbedMeshRef = useRef<THREE.Mesh | null>(null);
  const grabOffsetRef = useRef<THREE.Vector3>(new THREE.Vector3());
  const originalScaleRef = useRef<number>(1.0);
  const palmSmXRef = useRef<number | null>(null);
  const palmSmYRef = useRef<number | null>(null);
  const fistCountRef = useRef<number>(0);
  const prevIsFistRef = useRef<boolean>(false);
  const palmOpenStartTimeRef = useRef<number | null>(null);
  const extrudeTriggeredRef = useRef<boolean>(false);
  const lastColorSelectTimeRef = useRef<number>(0);

  const rightPinchSmoothedRef = useRef<number>(60);
  const rightPinchBaseRef = useRef<number>(60);
  const rightPinchScaleRef = useRef<number>(1.0);
  const rightPinchLastScaleRef = useRef<number>(1.0);
  const rightPinchLockedRef = useRef<boolean>(false);
  const suppressRightHandGesturesRef = useRef<boolean>(false);

  const stateRef = useRef<GestureState>("IDLE");
  const candidateGestureRef = useRef<string>("IDLE");
  const candidateCountRef = useRef<number>(0);

  const polaroidMeshRef = useRef<THREE.Mesh | null>(null);
  const openPolaroidModalRef = useRef<(mesh: THREE.Mesh) => void>(() => {});

  const polaroidExportBgRef = useRef(DEFAULT_POLAROID_EXPORT_BG);
  const polaroidAngleYRef = useRef(20);
  const polaroidAngleXRef = useRef(15);
  const polaroidNoteRef = useRef("");
  const polaroidPreviewGenRef = useRef(0);

  const [polaroidOpen, setPolaroidOpen] = useState(false);
  const [polaroidAngleY, setPolaroidAngleY] = useState(20);
  const [polaroidAngleX, setPolaroidAngleX] = useState(15);
  /** Polaroid export only — not wired to the live Three.js scene or page background. */
  const [polaroidExportBg, setPolaroidExportBg] = useState(
    DEFAULT_POLAROID_EXPORT_BG,
  );
  const [polaroidNoteInput, setPolaroidNoteInput] = useState("");
  const [polaroidPreview, setPolaroidPreview] = useState("");

  const schedulePolaroidPreview = useCallback(() => {
    const mesh = polaroidMeshRef.current;
    if (!mesh) return;
    const gen = ++polaroidPreviewGenRef.current;
    const render = renderShapeToDataURL(
      mesh,
      polaroidExportBgRef.current,
      polaroidAngleYRef.current,
      polaroidAngleXRef.current,
    );
    void buildPolaroid(
      render,
      polaroidNoteRef.current,
      polaroidExportBgRef.current,
    ).then((dataURL) => {
      if (gen !== polaroidPreviewGenRef.current) return;
      setPolaroidPreview(dataURL);
    });
  }, []);

  useEffect(() => {
    activeColorRef.current = activeColor;
  }, [activeColor]);

  useEffect(() => {
    if (!polaroidOpen) {
      polaroidPreviewGenRef.current++;
      return;
    }
    schedulePolaroidPreview();
  }, [polaroidOpen, schedulePolaroidPreview]);

  openPolaroidModalRef.current = (mesh: THREE.Mesh) => {
    polaroidMeshRef.current = mesh;
    polaroidExportBgRef.current = DEFAULT_POLAROID_EXPORT_BG;
    polaroidAngleYRef.current = 20;
    polaroidAngleXRef.current = 15;
    polaroidNoteRef.current = "";
    setPolaroidExportBg(DEFAULT_POLAROID_EXPORT_BG);
    setPolaroidAngleY(20);
    setPolaroidAngleX(15);
    setPolaroidNoteInput("");
    setPolaroidPreview("");
    setPolaroidOpen(true);
  };

  // Initialize Three.js
  useEffect(() => {
    if (!containerRef.current || !canvas3DRef.current) return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;

    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 2000);
    camera.position.z = 400;
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({
      canvas: canvas3DRef.current,
      alpha: true,
      antialias: true,
    });
    renderer.setClearColor(0x000000, 0);
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.42);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.35);
    dirLight.position.set(6, 10, 8);
    scene.add(dirLight);

    let animationFrameId: number;

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();

      // Gently rotate all shapes and handle bounce animation
      meshesRef.current.forEach((mesh) => {
        if (!polaroidSnapshotInProgress) {
          mesh.rotation.y += 0.005;
        }

        if (mesh.userData.bouncing) {
          const t = (performance.now() - mesh.userData.bounceStart) / 1000;
          const duration = 0.7;
          const overshoot = 1.3;
          let scale;
          if (t < duration) {
            const progress = t / duration;
            scale =
              overshoot *
                Math.sin(progress * Math.PI) *
                Math.pow(1 - progress, 0.4) +
              progress;
            scale = Math.max(0.01, Math.min(scale, overshoot));
          } else {
            scale = 1.0;
            mesh.userData.bouncing = false;
          }
          mesh.scale.setScalar(scale * mesh.userData.targetScale);
        }
      });

      renderer.render(scene, camera);
    };
    animate();

    const handleResize = () => {
      if (!containerRef.current) return;
      const newWidth = containerRef.current.clientWidth;
      const newHeight = containerRef.current.clientHeight;
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);

      if (canvas2DRef.current) {
        canvas2DRef.current.width = newWidth;
        canvas2DRef.current.height = newHeight;
      }
    };
    window.addEventListener("resize", handleResize);
    handleResize(); // Initial sizing for 2D canvas

    const CLICK_MAX_MOVE_PX = 10;
    const polaroidPointerDown = {
      active: false,
      x: 0,
      y: 0,
      pointerId: -1,
    };
    let primaryPointerDown = false;
    const canvasEl = renderer.domElement;

    const raycastMeshesAtClient = (clientX: number, clientY: number) => {
      const rect = canvasEl.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((clientX - rect.left) / rect.width) * 2 - 1,
        -((clientY - rect.top) / rect.height) * 2 + 1,
      );
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouse, camera);
      const targets = scene.children.filter(
        (obj): obj is THREE.Mesh =>
          obj instanceof THREE.Mesh && meshesRef.current.includes(obj),
      );
      return raycaster.intersectObjects(targets, false);
    };

    const updateShapeHoverCursor = (clientX: number, clientY: number) => {
      if (primaryPointerDown) return;
      const hits = raycastMeshesAtClient(clientX, clientY);
      canvasEl.style.cursor = hits.length > 0 ? "pointer" : "";
    };

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      primaryPointerDown = true;
      polaroidPointerDown.active = true;
      polaroidPointerDown.x = e.clientX;
      polaroidPointerDown.y = e.clientY;
      polaroidPointerDown.pointerId = e.pointerId;
    };

    const onPointerUp = (e: PointerEvent) => {
      if (e.button === 0) {
        primaryPointerDown = false;
        updateShapeHoverCursor(e.clientX, e.clientY);
      }

      if (
        !polaroidPointerDown.active ||
        e.pointerId !== polaroidPointerDown.pointerId
      )
        return;

      const sx = polaroidPointerDown.x;
      const sy = polaroidPointerDown.y;
      polaroidPointerDown.active = false;
      polaroidPointerDown.pointerId = -1;

      if (e.button !== 0) return;

      const dx = e.clientX - sx;
      const dy = e.clientY - sy;
      if (dx * dx + dy * dy > CLICK_MAX_MOVE_PX * CLICK_MAX_MOVE_PX) return;

      const hits = raycastMeshesAtClient(e.clientX, e.clientY);
      if (hits.length > 0) {
        const hit = hits[0].object;
        if (hit instanceof THREE.Mesh) {
          openPolaroidModalRef.current(hit);
        }
      }
    };

    const onPointerCancel = (e: PointerEvent) => {
      if (e.pointerId === polaroidPointerDown.pointerId) {
        polaroidPointerDown.active = false;
        polaroidPointerDown.pointerId = -1;
      }
      primaryPointerDown = false;
      canvasEl.style.cursor = "";
    };

    const onPointerMove = (e: PointerEvent) => {
      updateShapeHoverCursor(e.clientX, e.clientY);
    };

    const onPointerLeave = () => {
      canvasEl.style.cursor = "";
    };

    canvasEl.addEventListener("pointerdown", onPointerDown);
    canvasEl.addEventListener("pointerup", onPointerUp);
    canvasEl.addEventListener("pointercancel", onPointerCancel);
    canvasEl.addEventListener("pointermove", onPointerMove);
    canvasEl.addEventListener("pointerleave", onPointerLeave);

    return () => {
      canvasEl.removeEventListener("pointerdown", onPointerDown);
      canvasEl.removeEventListener("pointerup", onPointerUp);
      canvasEl.removeEventListener("pointercancel", onPointerCancel);
      canvasEl.removeEventListener("pointermove", onPointerMove);
      canvasEl.removeEventListener("pointerleave", onPointerLeave);
      canvasEl.style.cursor = "";
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animationFrameId);
      renderer.dispose();
    };
  }, []);

  const create3DShape = useCallback((points2D: THREE.Vector2[]) => {
    if (
      points2D.length < 3 ||
      !sceneRef.current ||
      !cameraRef.current ||
      !containerRef.current
    )
      return;

    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;
    const camera = cameraRef.current;

    const fov = camera.fov * (Math.PI / 180);
    const sceneHeight = 2 * Math.tan(fov / 2) * camera.position.z;
    const sceneWidth = sceneHeight * camera.aspect;

    const points3D = points2D.map((p) => {
      const worldX = (p.x / width) * sceneWidth - sceneWidth / 2;
      const worldY = -((p.y / height) * sceneHeight - sceneHeight / 2);
      return new THREE.Vector2(worldX, worldY);
    });

    // Close the path
    points3D.push(points3D[0].clone());

    const shape = new THREE.Shape(points3D);
    const extrudeSettings = {
      depth: 30,
      bevelEnabled: true,
      bevelThickness: 10,
      bevelSize: 8,
      bevelOffset: 0,
      bevelSegments: 12,
    };

    try {
      const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
      geometry.computeBoundingBox();
      const bboxCenter = new THREE.Vector3();
      geometry.boundingBox?.getCenter(bboxCenter);

      // Anchor to 2D stroke centroid (not 3D bbox center). Bevels shift the
      // bounding box, which pulled low strokes upward in Y while X looked fine.
      const loopPts = points3D.slice(0, -1);
      let cx = 0;
      let cy = 0;
      for (const p of loopPts) {
        cx += p.x;
        cy += p.y;
      }
      cx /= loopPts.length;
      cy /= loopPts.length;
      const anchor = new THREE.Vector3(cx, cy, bboxCenter.z);

      geometry.translate(-anchor.x, -anchor.y, -anchor.z);

      const material = new THREE.MeshStandardMaterial({
        color: activeColorRef.current,
        roughness: 0.25,
        metalness: 0.0,
        envMapIntensity: 1.0,
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.copy(anchor);
      mesh.rotation.x = Math.random() * 0.2 - 0.1;
      mesh.rotation.y = Math.random() * 0.2 - 0.1;
      mesh.rotation.z = Math.random() * 0.2 - 0.1;

      mesh.scale.setScalar(0.01);
      mesh.userData.bouncing = true;
      mesh.userData.bounceStart = performance.now();
      mesh.userData.targetScale = 1.0;

      sceneRef.current.add(mesh);
      meshesRef.current.push(mesh);
    } catch (e) {
      console.error(
        "Failed to extrude shape. Path might be self-intersecting.",
        e,
      );
    }
  }, []);

  const drawOn2DCanvas = useCallback(
    (
      points: THREE.Vector2[],
      rightPos: THREE.Vector2 | null,
      leftPos: THREE.Vector2 | null,
      palmOpenStartTime: number | null,
      extrudeTriggered: boolean,
      isFist: boolean,
      isGrabActive: boolean,
    ) => {
      const canvas = canvas2DRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (points.length > 0) {
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length - 1; i++) {
          const midX = (points[i].x + points[i + 1].x) / 2;
          const midY = (points[i].y + points[i + 1].y) / 2;
          ctx.quadraticCurveTo(points[i].x, points[i].y, midX, midY);
        }
        if (points.length > 1) {
          const last = points[points.length - 1];
          ctx.lineTo(last.x, last.y);
        }
        ctx.strokeStyle = activeColorRef.current;
        ctx.lineWidth = 3;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.stroke();
      }

      // Draw right cursor
      if (rightPos) {
        if (palmOpenStartTime && !extrudeTriggered) {
          const elapsed = performance.now() - palmOpenStartTime;
          const progress = Math.min(elapsed / FINISH_PALM_HOLD_MS, 1);
          const radius = 28;
          const startAngle = -Math.PI / 2;
          const endAngle = startAngle + progress * Math.PI * 2;

          ctx.beginPath();
          ctx.arc(rightPos.x, rightPos.y, radius, 0, Math.PI * 2);
          ctx.strokeStyle = "rgba(255,255,255,0.15)";
          ctx.lineWidth = 3;
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(rightPos.x, rightPos.y, radius, startAngle, endAngle);
          ctx.strokeStyle = activeColorRef.current;
          ctx.lineWidth = 3;
          ctx.lineCap = "round";
          ctx.stroke();
        }

        ctx.beginPath();
        ctx.arc(rightPos.x, rightPos.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = activeColorRef.current;
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Draw left cursor
      if (leftPos) {
        ctx.beginPath();
        ctx.arc(leftPos.x, leftPos.y, 6, 0, Math.PI * 2);

        if (isFist && isGrabActive) {
          ctx.fillStyle = "white";
          ctx.fill();
          ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
          ctx.lineWidth = 2;
          ctx.stroke();

          ctx.fillStyle = "white";
          ctx.font = "11px sans-serif";
          ctx.fillText("holding", leftPos.x - 20, leftPos.y + 20);
        } else if (isFist && !isGrabActive) {
          const pulse = 0.4 + 0.4 * ((Math.sin(Date.now() / 150) + 1) / 2);
          ctx.strokeStyle = `rgba(255, 255, 255, ${pulse})`;
          ctx.lineWidth = 2;
          ctx.stroke();
        } else {
          ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
    },
    [],
  );

  const clear2DCanvas = useCallback(() => {
    const canvas = canvas2DRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, []);

  const clearCanvas = useCallback(() => {
    const scene = sceneRef.current;
    if (scene) {
      meshesRef.current.forEach((mesh) => {
        scene.remove(mesh);
        mesh.geometry.dispose();
        const mat = mesh.material as THREE.MeshStandardMaterial;
        mat.dispose();
      });
    }
    meshesRef.current = [];

    if (grabbedMeshRef.current) {
      const mat = grabbedMeshRef.current.material as THREE.MeshStandardMaterial;
      mat.emissive.setHex(0x000000);
    }
    grabbedMeshRef.current = null;
    isGrabActiveRef.current = false;
    suppressRightHandGesturesRef.current = false;
    rightPinchLockedRef.current = false;
    rightPinchSmoothedRef.current = 60;

    currentPointsRef.current = [];
    clear2DCanvas();
    stateRef.current = "IDLE";
    palmOpenStartTimeRef.current = null;
    extrudeTriggeredRef.current = false;
    candidateGestureRef.current = "IDLE";
    candidateCountRef.current = 0;
  }, [clear2DCanvas]);

  // Initialize MediaPipe
  useEffect(() => {
    if (!videoRef.current) return;
    let isUnmounted = false;

    const hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      },
    });

    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7,
    });

    hands.onResults((results: Results) => {
      if (isUnmounted) return;
      setIsModelLoading(false);

      if (!containerRef.current) return;
      const width = containerRef.current.clientWidth;
      const height = containerRef.current.clientHeight;

      let leftLandmarks: any = null;
      let rightLandmarks: any = null;

      if (results.multiHandLandmarks && results.multiHandedness) {
        for (let i = 0; i < results.multiHandLandmarks.length; i++) {
          const classification = results.multiHandedness[i].label;
          if (classification === "Right") {
            leftLandmarks = results.multiHandLandmarks[i];
          } else if (classification === "Left") {
            rightLandmarks = results.multiHandLandmarks[i];
          }
        }
      }

      const getMappedPos = (lm: any) => {
        const rawX = 1 - lm.x;
        const rawY = lm.y;
        const mappedX = rawX * width;
        const mappedY = rawY * height;
        return new THREE.Vector2(mappedX, mappedY);
      };

      // --- SKELETON PREVIEW ---
      if (previewCanvasRef.current && videoRef.current) {
        const pCanvas = previewCanvasRef.current;
        const pCtx = pCanvas.getContext("2d");
        if (pCtx) {
          const pw = previewLayoutRef.current.w;
          const ph = previewLayoutRef.current.h;
          pCanvas.width = pw;
          pCanvas.height = ph;
          pCtx.clearRect(0, 0, pCanvas.width, pCanvas.height);

          pCtx.save();
          pCtx.translate(pCanvas.width, 0);
          pCtx.scale(-1, 1);

          if (
            results.multiHandLandmarks &&
            results.multiHandLandmarks.length > 0
          ) {
            for (let i = 0; i < results.multiHandLandmarks.length; i++) {
              const landmarks = results.multiHandLandmarks[i];
              const classification = results.multiHandedness[i].label;
              // User's left hand (MediaPipe "Right") = green; user's right ("Left") = red
              const color = classification === "Right" ? "#22c55e" : "#ef4444";

              pCtx.strokeStyle = color;
              pCtx.lineWidth = Math.max(1.5, pw / 200);
              const connections = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 4],
                [0, 5],
                [5, 6],
                [6, 7],
                [7, 8],
                [5, 9],
                [9, 10],
                [10, 11],
                [11, 12],
                [9, 13],
                [13, 14],
                [14, 15],
                [15, 16],
                [13, 17],
                [17, 18],
                [18, 19],
                [19, 20],
                [0, 17],
              ];

              connections.forEach(([a, b]) => {
                const p1 = landmarks[a];
                const p2 = landmarks[b];
                pCtx.beginPath();
                pCtx.moveTo(p1.x * pw, p1.y * ph);
                pCtx.lineTo(p2.x * pw, p2.y * ph);
                pCtx.stroke();
              });

              pCtx.fillStyle = color;
              const jointR = Math.max(2, pw / 120);
              landmarks.forEach((lm: any) => {
                pCtx.beginPath();
                pCtx.arc(lm.x * pw, lm.y * ph, jointR, 0, 2 * Math.PI);
                pCtx.fill();
              });
            }
          }
          pCtx.restore();
        }
      }

      if (!leftLandmarks && !rightLandmarks) {
        drawOn2DCanvas(
          currentPointsRef.current,
          null,
          null,
          palmOpenStartTimeRef.current,
          extrudeTriggeredRef.current,
          false,
          false,
        );
        smoothedCursorRef.current = null;
        fistCountRef.current = 0;
        prevIsFistRef.current = false;
        palmSmXRef.current = null;
        palmSmYRef.current = null;

        // End grab if left hand lost
        if (isGrabActiveRef.current && grabbedMeshRef.current) {
          const mat = grabbedMeshRef.current
            .material as THREE.MeshStandardMaterial;
          mat.emissive.setHex(0x000000);
          grabbedMeshRef.current = null;
          isGrabActiveRef.current = false;
        }
        return;
      }

      // --- LEFT HAND LOGIC ---
      let leftPalmPos: THREE.Vector2 | null = null;
      let isFist = false;
      if (!leftLandmarks) {
        fistCountRef.current = 0;
        prevIsFistRef.current = false;
        palmSmXRef.current = null;
        palmSmYRef.current = null;
        if (isGrabActiveRef.current && grabbedMeshRef.current) {
          const mat = grabbedMeshRef.current
            .material as THREE.MeshStandardMaterial;
          mat.emissive.setHex(0x000000);
        }
        grabbedMeshRef.current = null;
        isGrabActiveRef.current = false;
      } else {
        const width = containerRef.current.clientWidth;
        const height = containerRef.current.clientHeight;

        const palmCenter = (lm: any) => {
          const pts = [lm[0], lm[5], lm[9], lm[13], lm[17]];
          let nx = 0,
            ny = 0;
          pts.forEach((p) => {
            nx += 1.0 - p.x;
            ny += p.y;
          });
          nx /= pts.length;
          ny /= pts.length;

          const sx = nx * width;
          const sy = ny * height;

          if (palmSmXRef.current === null || palmSmYRef.current === null) {
            palmSmXRef.current = sx;
            palmSmYRef.current = sy;
          } else {
            palmSmXRef.current += 0.35 * (sx - palmSmXRef.current);
            palmSmYRef.current += 0.35 * (sy - palmSmYRef.current);
          }
          return new THREE.Vector2(palmSmXRef.current, palmSmYRef.current);
        };

        const detectFist = (lm: any) => {
          return (
            lm[8].y > lm[6].y &&
            lm[12].y > lm[10].y &&
            lm[16].y > lm[14].y &&
            lm[20].y > lm[18].y
          );
        };

        const detectOpen = (lm: any) => {
          return (
            lm[8].y < lm[6].y - 0.04 &&
            lm[12].y < lm[10].y - 0.04 &&
            lm[16].y < lm[14].y - 0.04 &&
            lm[20].y < lm[18].y - 0.04
          );
        };

        const lmDist = (a: any, b: any) => {
          const ax = (1.0 - a.x) * width;
          const ay = a.y * height;
          const bx = (1.0 - b.x) * width;
          const by = b.y * height;
          return Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2);
        };

        leftPalmPos = palmCenter(leftLandmarks);

        const rawFist = detectFist(leftLandmarks);

        fistCountRef.current = rawFist
          ? Math.min(fistCountRef.current + 1, 12)
          : Math.max(fistCountRef.current - 1, 0);

        isFist = fistCountRef.current >= 7;

        const fistRising = isFist && !prevIsFistRef.current;
        const fistFalling = !isFist && prevIsFistRef.current;
        prevIsFistRef.current = isFist;

        if (fistRising && !isGrabActiveRef.current) {
          const ndcX = (leftPalmPos.x / width) * 2 - 1;
          const ndcY = -(leftPalmPos.y / height) * 2 + 1;
          const pw = new THREE.Vector3(ndcX, ndcY, 0.5);
          pw.unproject(cameraRef.current!);
          const dir = pw.sub(cameraRef.current!.position).normalize();
          const distance = -cameraRef.current!.position.z / dir.z;
          const targetPos = cameraRef
            .current!.position.clone()
            .add(dir.multiplyScalar(distance));

          let closestMesh: THREE.Mesh | null = null;
          let minDist = 100;

          meshesRef.current.forEach((mesh) => {
            const box = new THREE.Box3().setFromObject(mesh);
            const center = new THREE.Vector3();
            box.getCenter(center);
            const dist = center.distanceTo(targetPos);
            if (dist < minDist) {
              minDist = dist;
              closestMesh = mesh;
            }
          });

          if (closestMesh) {
            grabbedMeshRef.current = closestMesh;
            grabOffsetRef.current.copy(closestMesh.position).sub(targetPos);
            originalScaleRef.current = closestMesh.scale.x;
            isGrabActiveRef.current = true;

            const mat = closestMesh.material as THREE.MeshStandardMaterial;
            mat.emissive.set(mat.color).multiplyScalar(0.2);
          }
        }

        if (isGrabActiveRef.current && grabbedMeshRef.current && isFist) {
          const ndcX = (leftPalmPos.x / width) * 2 - 1;
          const ndcY = -(leftPalmPos.y / height) * 2 + 1;
          const pw = new THREE.Vector3(ndcX, ndcY, 0.5);
          pw.unproject(cameraRef.current!);
          const dir = pw.sub(cameraRef.current!.position).normalize();
          const distance = -cameraRef.current!.position.z / dir.z;
          const targetPos = cameraRef
            .current!.position.clone()
            .add(dir.multiplyScalar(distance));

          grabbedMeshRef.current.position.x =
            targetPos.x + grabOffsetRef.current.x;
          grabbedMeshRef.current.position.y =
            targetPos.y + grabOffsetRef.current.y;
        }

        if (fistFalling && isGrabActiveRef.current) {
          if (grabbedMeshRef.current) {
            const mat = grabbedMeshRef.current
              .material as THREE.MeshStandardMaterial;
            mat.emissive.setHex(0x000000);
          }
          grabbedMeshRef.current = null;
          isGrabActiveRef.current = false;

          suppressRightHandGesturesRef.current = false;
          rightPinchLockedRef.current = false;
          rightPinchSmoothedRef.current = 60;
        }
      }

      // --- TWO-HAND RESIZE LOGIC ---
      if (
        isFist &&
        isGrabActiveRef.current &&
        grabbedMeshRef.current &&
        rightLandmarks
      ) {
        const width = containerRef.current.clientWidth;
        const height = containerRef.current.clientHeight;
        const lmDist = (a: any, b: any) => {
          const ax = (1.0 - a.x) * width;
          const ay = a.y * height;
          const bx = (1.0 - b.x) * width;
          const by = b.y * height;
          return Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2);
        };

        const rawRightPinch = lmDist(rightLandmarks[4], rightLandmarks[8]);
        rightPinchSmoothedRef.current =
          rightPinchSmoothedRef.current * 0.8 + rawRightPinch * 0.2;

        if (!rightPinchLockedRef.current) {
          rightPinchBaseRef.current = rightPinchSmoothedRef.current;
          rightPinchScaleRef.current = 1.0;
          rightPinchLastScaleRef.current = 1.0;
          rightPinchLockedRef.current = true;
          originalScaleRef.current = grabbedMeshRef.current.scale.x;
        }

        const sf = Math.max(
          0.15,
          Math.min(
            rightPinchSmoothedRef.current / rightPinchBaseRef.current,
            5.0,
          ),
        );
        rightPinchScaleRef.current =
          rightPinchScaleRef.current * 0.85 + sf * 0.15;

        if (
          Math.abs(
            rightPinchScaleRef.current - rightPinchLastScaleRef.current,
          ) > 0.02
        ) {
          grabbedMeshRef.current.scale.setScalar(
            originalScaleRef.current * rightPinchScaleRef.current,
          );
          rightPinchLastScaleRef.current = rightPinchScaleRef.current;
        }

        suppressRightHandGesturesRef.current = true;
      } else {
        suppressRightHandGesturesRef.current = false;
        rightPinchLockedRef.current = false;
        rightPinchSmoothedRef.current = 60;
      }

      // --- RIGHT HAND LOGIC ---
      let rightIndexPos: THREE.Vector2 | null = null;
      if (rightLandmarks) {
        const rawRightIndexTip = getMappedPos(rightLandmarks[8]);
        if (!smoothedCursorRef.current) {
          smoothedCursorRef.current = rawRightIndexTip.clone();
        } else {
          smoothedCursorRef.current.x +=
            0.35 * (rawRightIndexTip.x - smoothedCursorRef.current.x);
          smoothedCursorRef.current.y +=
            0.35 * (rawRightIndexTip.y - smoothedCursorRef.current.y);
        }
        rightIndexPos = smoothedCursorRef.current.clone();

        const thumbTip = getMappedPos(rightLandmarks[4]);

        const containerRect = containerRef.current!.getBoundingClientRect();
        const rawPinchDist = rawRightIndexTip.distanceTo(thumbTip);
        // Pinch-to-pick uses the same "pointing" finger pose as draw; keep draw off while thumb and index are close.
        const PINCH_SUPPRESS_DRAW_PX = 72;
        const paletteEl = document.getElementById("color-palette");
        let indexOverPalette = false;
        if (paletteEl) {
          const pr = paletteEl.getBoundingClientRect();
          const pad = 32;
          const ix = containerRect.left + rawRightIndexTip.x;
          const iy = containerRect.top + rawRightIndexTip.y;
          indexOverPalette =
            ix >= pr.left - pad &&
            ix <= pr.right + pad &&
            iy >= pr.top - pad &&
            iy <= pr.bottom + pad;
        }
        const suppressDrawForColorPick =
          rawPinchDist < PINCH_SUPPRESS_DRAW_PX || indexOverPalette;

        if (!suppressRightHandGesturesRef.current) {
          const r = rightLandmarks;
          const indexExt = r[8].y < r[6].y - RIGHT_FINGER_EXTEND_NORM;
          const middleExt = r[12].y < r[10].y - RIGHT_FINGER_EXTEND_NORM;
          const ringExt = r[16].y < r[14].y - RIGHT_FINGER_EXTEND_NORM;
          const pinkyExt = r[20].y < r[18].y - RIGHT_FINGER_EXTEND_NORM;
          const thumbExtended =
            Math.abs(r[3].x - r[4].x) > RIGHT_THUMB_SPREAD_NORM;

          let detectedGesture = "IDLE";

          if (
            !suppressDrawForColorPick &&
            indexExt &&
            !middleExt &&
            !ringExt &&
            !pinkyExt
          ) {
            detectedGesture = "DRAW";
          } else if (
            indexExt &&
            middleExt &&
            ringExt &&
            pinkyExt &&
            thumbExtended
          ) {
            detectedGesture = "FINISH";
          }

          if (detectedGesture === "FINISH" && stateRef.current === "DRAW") {
            if (!palmOpenStartTimeRef.current) {
              palmOpenStartTimeRef.current = performance.now();
            } else {
              const elapsed = performance.now() - palmOpenStartTimeRef.current;
              if (
                elapsed >= FINISH_PALM_HOLD_MS &&
                !extrudeTriggeredRef.current &&
                currentPointsRef.current.length > 3
              ) {
                extrudeTriggeredRef.current = true;
                create3DShape(currentPointsRef.current);
                currentPointsRef.current = [];
                clear2DCanvas();
                palmOpenStartTimeRef.current = null;
                extrudeTriggeredRef.current = false;
                stateRef.current = "IDLE";
              }
            }
          } else {
            palmOpenStartTimeRef.current = null;
          }

          // Color Selection (viewport coords — mapped points are container-local)
          const pinchDist = rightIndexPos.distanceTo(thumbTip);
          const isRightPinching = pinchDist < 56;

          if (isRightPinching) {
            const now = performance.now();
            if (now - lastColorSelectTimeRef.current > 300) {
              const swatches = document.querySelectorAll('[id^="swatch-"]');
              const pickX = containerRect.left + rightIndexPos.x;
              const pickY = containerRect.top + rightIndexPos.y;
              let hoveredSwatch: Element | null = null;
              swatches.forEach((swatch) => {
                const rect = swatch.getBoundingClientRect();
                const centerX = rect.left + rect.width / 2;
                const centerY = rect.top + rect.height / 2;
                const dist = Math.hypot(pickX - centerX, pickY - centerY);
                if (dist <= 44) {
                  hoveredSwatch = swatch;
                }
              });

              if (hoveredSwatch) {
                const colorName = hoveredSwatch.id.replace("swatch-", "");
                const colorObj = COLORS.find((c) => c.name === colorName);
                if (colorObj) {
                  lastColorSelectTimeRef.current = now;
                  if (activeColorRef.current !== colorObj.value) {
                    setActiveColor(colorObj.value);
                  }
                  setPinchedColor(colorObj.value);
                  setTimeout(() => setPinchedColor(null), 400);
                  if (stateRef.current === "DRAW") {
                    currentPointsRef.current = [];
                    clear2DCanvas();
                    stateRef.current = "IDLE";
                    candidateGestureRef.current = "IDLE";
                    candidateCountRef.current = 0;
                  }
                }
              }
            }
          }

          if (detectedGesture !== "FINISH") {
            if (detectedGesture === candidateGestureRef.current) {
              candidateCountRef.current += 1;
            } else {
              candidateGestureRef.current = detectedGesture;
              candidateCountRef.current = 1;
            }

            if (
              candidateCountRef.current >= 5 &&
              stateRef.current !== candidateGestureRef.current
            ) {
              const prevState = stateRef.current;
              stateRef.current = candidateGestureRef.current as any;

              if (stateRef.current === "DRAW") {
                if (prevState === "IDLE") {
                  currentPointsRef.current = [];
                }
              }
            }
          }

          // Handle Drawing
          if (stateRef.current === "DRAW" && detectedGesture === "DRAW") {
            const points = currentPointsRef.current;
            if (points.length === 0) {
              points.push(rightIndexPos);
            } else {
              const lastPoint = points[points.length - 1];
              const dx = rightIndexPos.x - lastPoint.x;
              const dy = rightIndexPos.y - lastPoint.y;
              if (Math.sqrt(dx * dx + dy * dy) > 4) {
                points.push(rightIndexPos);
              }
            }
          }
        }
        drawOn2DCanvas(
          currentPointsRef.current,
          rightIndexPos,
          leftPalmPos,
          palmOpenStartTimeRef.current,
          extrudeTriggeredRef.current,
          isFist,
          isGrabActiveRef.current,
        );
      } else {
        smoothedCursorRef.current = null;
        drawOn2DCanvas(
          currentPointsRef.current,
          null,
          leftPalmPos,
          palmOpenStartTimeRef.current,
          extrudeTriggeredRef.current,
          isFist,
          isGrabActiveRef.current,
        );
      }
    });

    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        if (videoRef.current && !isUnmounted) {
          try {
            await hands.send({ image: videoRef.current });
          } catch (e) {
            console.error("MediaPipe send error:", e);
          }
        }
      },
      width: 1280,
      height: 720,
    });
    camera.start();

    return () => {
      isUnmounted = true;
      camera.stop();
      try {
        hands.close();
      } catch (e) {
        console.error("Error closing hands:", e);
      }
    };
  }, [create3DShape, drawOn2DCanvas, clear2DCanvas]);

  const instructionChrome = useMemo(() => {
    switch (previewLayout.tier) {
      case "tablet":
        return {
          edge: 16,
          pad: "8px 12px",
          minW: 140,
          titleFs: 10,
          headFs: 11,
          bodyFs: 10,
          titleMb: 8,
          blockMb: 4,
          cameraTop: 12,
        };
      case "mobile":
        return {
          edge: 8,
          pad: "5px 8px",
          minW: 116,
          titleFs: 9,
          headFs: 10,
          bodyFs: 9,
          titleMb: 6,
          blockMb: 3,
          cameraTop: 16,
        };
      default:
        return {
          edge: 20,
          pad: "12px 16px",
          minW: 160,
          titleFs: 11,
          headFs: 12,
          bodyFs: 11,
          titleMb: 10,
          blockMb: 6,
          cameraTop: 16,
        };
    }
  }, [previewLayout.tier]);

  const paletteIconSize =
    previewLayout.tier === "desktop"
      ? 20
      : previewLayout.tier === "tablet"
        ? 18
        : 16;

  return (
    <div className="flex h-screen w-full overflow-hidden font-sans text-neutral-100 relative">
      {/* Background Gradient */}
      <div
        style={{
          position: "fixed",
          width: "100%",
          height: "100%",
          zIndex: -1,
          background:
            "radial-gradient(ellipse at 50% 40%, #1c1c1c 0%, #111111 55%, #080808 100%)",
        }}
      />

      {/* Right Canvas Area */}
      <div className="flex-1 relative" ref={containerRef}>
        <div className="fixed top-4 right-4 z-50 flex max-sm:max-w-[min(52vw,11rem)] max-sm:shrink flex-col items-end gap-2">
          <button
            type="button"
            onClick={clearCanvas}
            className="cursor-pointer rounded-lg border border-neutral-700 bg-neutral-900/85 px-3 py-2 text-xs font-regular text-neutral-100 shadow-lg backdrop-blur-md transition-colors hover:bg-neutral-800/90 max-sm:px-2 max-sm:py-1.5 max-sm:text-[10px]"
          >
            CLEAR CANVAS
          </button>
          <div className="flex flex-col items-end gap-0.5 text-right text-[11px] leading-snug text-neutral-400 max-sm:text-[9px] max-sm:leading-tight">
            <span>SCROLL to zoom</span>
            <span>DRAG to rotate</span>
            {previewLayout.tier === "desktop" ? (
              <>
                <span>RIGHT-CLICK to pan</span>
                <span>LEFT-CLICK to share object</span>
              </>
            ) : (
              <span>TAP to share object</span>
            )}
          </div>
        </div>

        {/* 3D Canvas */}
        <canvas
          ref={canvas3DRef}
          className="absolute inset-0 w-full h-full outline-none"
        />

        {/* 2D Overlay Canvas */}
        <canvas
          ref={canvas2DRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />

        {/* Polaroid export modal — full-bleed absolute inside canvas area (not fixed) */}
        <div
          id="polaroid-overlay"
          role="presentation"
          onClick={(e) => {
            if (e.target === e.currentTarget) {
              setPolaroidOpen(false);
              polaroidMeshRef.current = null;
            }
          }}
          className="min-h-full flex-col items-center justify-center overflow-y-auto overscroll-contain p-4 max-lg:box-border max-lg:px-4 max-lg:py-8 max-sm:px-3.5 max-sm:py-10"
          style={{
            display: polaroidOpen ? "flex" : "none",
            position: "absolute",
            inset: 0,
            zIndex: 80,
            alignItems: "center",
            justifyContent: "center",
            background: "rgba(0,0,0,0.55)",
            backdropFilter: "blur(6px)",
          }}
        >
          <div
            role="dialog"
            aria-modal="true"
            aria-labelledby="polaroid-modal-title"
            onClick={(e) => e.stopPropagation()}
            className="flex min-h-0 w-[min(92vw,420px)] flex-col overflow-y-auto overflow-x-hidden overscroll-contain rounded-xl border border-white/15 bg-neutral-900/92 p-5 shadow-2xl backdrop-blur-md max-h-[min(92vh,900px)] max-lg:max-h-[calc(100vh-3.25rem)] max-lg:min-h-[calc(100vh-3.25rem)] max-lg:w-[min(86vw,336px)] max-lg:p-3 max-sm:max-h-[calc(100vh-5.25rem)] max-sm:min-h-[calc(100vh-5.25rem)] max-sm:w-[min(82vw,292px)] max-sm:p-2.5 supports-[height:100dvh]:max-lg:max-h-[calc(100dvh-3.25rem)] supports-[height:100dvh]:max-lg:min-h-[calc(100dvh-3.25rem)] supports-[height:100dvh]:max-sm:max-h-[calc(100dvh-5.25rem)] supports-[height:100dvh]:max-sm:min-h-[calc(100dvh-5.25rem)]"
          >
            <div className="mb-4 flex shrink-0 items-center justify-between gap-3 max-lg:mb-2.5 max-lg:gap-2 max-sm:mb-2">
              <h2
                id="polaroid-modal-title"
                className="m-0 text-base font-semibold leading-snug text-white max-lg:text-sm max-sm:text-[13px]"
              >
                Share The Love
              </h2>
              <button
                type="button"
                id="close-modal-btn"
                aria-label="Close"
                onClick={() => {
                  setPolaroidOpen(false);
                  polaroidMeshRef.current = null;
                }}
                className="flex size-9 shrink-0 items-center justify-center rounded-md text-white/60 transition-colors hover:bg-white/10 hover:text-white max-lg:size-8 max-sm:size-7"
              >
                <X size={18} strokeWidth={2} aria-hidden />
              </button>
            </div>

            <div className="flex min-h-0 flex-col lg:contents max-lg:min-h-0 max-lg:flex-1 max-lg:gap-2 max-sm:gap-3">
              <div className="mb-4 flex min-h-0 items-center justify-center max-lg:mb-0 max-lg:shrink-0 max-lg:justify-center max-lg:py-3 max-sm:mb-0 max-sm:shrink-0 max-sm:justify-center max-sm:py-5 lg:mb-4 lg:flex-none">
                {polaroidPreview ? (
                  <img
                    id="polaroid-preview"
                    src={polaroidPreview}
                    alt="Polaroid preview"
                    className="mx-auto block w-full max-w-[280px] rounded-t-md object-contain shadow-lg max-lg:h-auto max-lg:w-auto max-lg:max-w-[min(240px,78vw)] max-lg:max-h-[min(38dvh,260px)] max-sm:max-h-[min(34dvh,232px)] max-sm:max-w-[min(220px,72vw)]"
                  />
                ) : (
                  <div className="mx-auto flex h-[320px] w-full max-w-[280px] items-center justify-center rounded-t-md bg-black/30 px-2 text-sm leading-snug text-white/40 max-lg:h-28 max-lg:min-h-28 max-lg:max-h-28 max-lg:max-w-[min(240px,78vw)] max-lg:text-xs max-lg:leading-snug max-sm:h-24 max-sm:min-h-24 max-sm:max-h-24 max-sm:max-w-[min(220px,72vw)] max-sm:px-1.5 max-sm:text-[11px]">
                    Generating preview…
                  </div>
                )}
              </div>

              <div className="shrink-0 space-y-0 lg:space-y-0 max-lg:flex max-lg:min-h-0 max-lg:flex-1 max-lg:flex-col max-lg:py-1 max-sm:py-0 max-sm:pb-1">
                <div className="flex shrink-0 flex-col gap-5 max-sm:gap-4 lg:contents">
                  <div className="mb-3 max-lg:mb-0 max-sm:mb-0">
                    <div className="mb-2 text-[11px] font-medium uppercase tracking-wide text-white/40 max-lg:mb-1.5 max-lg:text-[10px] max-lg:tracking-[0.14em] max-sm:mb-1 max-sm:text-[9px] max-sm:leading-tight">
                      VIEW
                    </div>
                    <div className="space-y-3 max-lg:space-y-2.5 max-sm:space-y-2">
                      <div>
                        <div className="mb-1.5 flex justify-between text-[12px] leading-snug text-white/70 max-lg:mb-1.5 max-lg:text-[12px] max-sm:mb-1 max-sm:text-[11px]">
                          <span>LEFT/ RIGHT</span>
                          <span
                            id="angle-y-val"
                            className="tabular-nums text-white/85"
                          >
                            {polaroidAngleY}°
                          </span>
                        </div>
                        <input
                          id="angle-y"
                          type="range"
                          min={-180}
                          max={180}
                          value={polaroidAngleY}
                          onChange={(e) => {
                            const v = Number(e.target.value);
                            polaroidAngleYRef.current = v;
                            setPolaroidAngleY(v);
                          }}
                          onPointerUp={schedulePolaroidPreview}
                          onKeyUp={(e) => {
                            if (
                              e.key === "ArrowLeft" ||
                              e.key === "ArrowRight" ||
                              e.key === "ArrowUp" ||
                              e.key === "ArrowDown" ||
                              e.key === "Home" ||
                              e.key === "End"
                            ) {
                              schedulePolaroidPreview();
                            }
                          }}
                          className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-white max-lg:mt-0.5 max-lg:h-2 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white max-sm:h-1.5 max-sm:[&::-webkit-slider-thumb]:h-2.5 max-sm:[&::-webkit-slider-thumb]:w-2.5"
                        />
                      </div>
                      <div>
                        <div className="mb-1.5 flex justify-between text-[12px] leading-snug text-white/70 max-lg:mb-1.5 max-lg:text-[12px] max-sm:mb-1 max-sm:text-[11px]">
                          <span>UP/ DOWN</span>
                          <span
                            id="angle-x-val"
                            className="tabular-nums text-white/85"
                          >
                            {polaroidAngleX}°
                          </span>
                        </div>
                        <input
                          id="angle-x"
                          type="range"
                          min={-90}
                          max={90}
                          value={polaroidAngleX}
                          onChange={(e) => {
                            const v = Number(e.target.value);
                            polaroidAngleXRef.current = v;
                            setPolaroidAngleX(v);
                          }}
                          onPointerUp={schedulePolaroidPreview}
                          onKeyUp={(e) => {
                            if (
                              e.key === "ArrowLeft" ||
                              e.key === "ArrowRight" ||
                              e.key === "ArrowUp" ||
                              e.key === "ArrowDown" ||
                              e.key === "Home" ||
                              e.key === "End"
                            ) {
                              schedulePolaroidPreview();
                            }
                          }}
                          className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-white max-lg:mt-0.5 max-lg:h-2 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white max-sm:h-1.5 max-sm:[&::-webkit-slider-thumb]:h-2.5 max-sm:[&::-webkit-slider-thumb]:w-2.5"
                        />
                      </div>
                    </div>
                  </div>

                  <div className="mb-3 max-lg:mb-0 max-sm:mb-0">
                    <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-white/40 max-lg:mb-1.5 max-lg:text-[10px] max-lg:tracking-[0.14em] max-sm:mb-1 max-sm:text-[9px] max-sm:leading-tight">
                      BACKGROUND COLOR
                    </div>
                    <div className="flex flex-wrap gap-2 max-lg:gap-2 max-sm:gap-1.5">
                      {POLAROID_BG_SWATCHES.map((c, i) => (
                        <button
                          key={c.value}
                          type="button"
                          className={`bg-swatch ${polaroidExportBg === c.value ? "active-bg" : ""}`}
                          data-color={c.value}
                          style={{ backgroundColor: c.value }}
                          title={`${c.name} · ${i < POLAROID_BG_DARK_COUNT ? "dark" : "light"}`}
                          onClick={() => {
                            polaroidExportBgRef.current = c.value;
                            setPolaroidExportBg(c.value);
                            schedulePolaroidPreview();
                          }}
                        />
                      ))}
                    </div>
                  </div>

                  <label className="mb-4 block max-lg:mb-0 max-sm:mb-0">
                    <div className="mb-1.5 flex items-baseline justify-between gap-2 max-lg:mb-1.5 max-sm:mb-1">
                      <span className="text-[11px] font-medium uppercase tracking-wide text-white/40 max-lg:text-[10px] max-lg:tracking-[0.14em] max-sm:text-[9px] max-sm:leading-tight">
                        CAPTION
                      </span>
                      <span
                        className={`tabular-nums text-[11px] max-lg:text-[10px] max-sm:text-[9px] ${
                          POLAROID_NOTE_MAX_CHARS - polaroidNoteInput.length ===
                          0
                            ? "text-red-400/90"
                            : POLAROID_NOTE_MAX_CHARS -
                                  polaroidNoteInput.length <=
                                10
                              ? "text-orange-400/90"
                              : "text-white/50"
                        }`}
                        aria-live="polite"
                        aria-label={`${POLAROID_NOTE_MAX_CHARS - polaroidNoteInput.length} characters remaining of ${POLAROID_NOTE_MAX_CHARS}`}
                      >
                        {POLAROID_NOTE_MAX_CHARS - polaroidNoteInput.length}
                      </span>
                    </div>
                    <textarea
                      id="polaroid-note"
                      value={polaroidNoteInput}
                      onChange={(e) => {
                        const v = e.target.value;
                        polaroidNoteRef.current = v;
                        setPolaroidNoteInput(v);
                        schedulePolaroidPreview();
                      }}
                      rows={2}
                      maxLength={POLAROID_NOTE_MAX_CHARS}
                      className="w-full resize-none rounded-md border border-white/12 bg-black/25 px-3 py-2 text-sm leading-relaxed text-white placeholder:text-white/30 outline-none focus:border-white/25 max-lg:px-2.5 max-lg:py-2 max-lg:text-[12px] max-lg:leading-snug max-sm:px-2 max-sm:py-1.5 max-sm:text-[11px] max-sm:leading-snug"
                      placeholder="my little note"
                    />
                  </label>
                </div>

                <button
                  type="button"
                  id="save-png-btn"
                  onClick={async () => {
                    const mesh = polaroidMeshRef.current;
                    if (!mesh) return;
                    const angleY = polaroidAngleYRef.current;
                    const angleX = polaroidAngleXRef.current;
                    const bg = polaroidExportBgRef.current;
                    const render = renderShapeToDataURL(
                      mesh,
                      bg,
                      angleY,
                      angleX,
                    );
                    const dataURL = await buildPolaroid(
                      render,
                      polaroidNoteInput,
                      bg,
                    );
                    const a = document.createElement("a");
                    a.href = dataURL;
                    a.download = "art.png";
                    a.click();
                  }}
                  className="mt-auto w-full shrink-0 rounded-lg bg-white/90 py-2.5 text-sm font-medium leading-snug text-neutral-900 transition-colors hover:bg-white max-lg:py-2 max-lg:text-xs max-sm:py-2 max-sm:text-[11px] lg:mt-0"
                >
                  SAVE PNG
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right Palette (Overlay) */}
        <div
          id="color-palette"
          className="absolute right-4 top-1/2 z-10 flex -translate-y-1/2 flex-col items-center gap-3 rounded-lg border border-neutral-800 bg-neutral-900/80 px-2 py-4 shadow-xl backdrop-blur-md max-lg:right-2 max-lg:gap-2 max-lg:px-1.5 max-lg:py-2 max-sm:gap-1.5 max-sm:py-1.5"
        >
          <div className="mb-2 text-neutral-400 max-lg:mb-1 max-sm:mb-0">
            <Palette size={paletteIconSize} />
          </div>
          {COLORS.map((color) => {
            const isWhite = color.name === "white";
            const selectedRing = isWhite
              ? "ring-2 ring-neutral-400 ring-offset-2 ring-offset-neutral-900 scale-110"
              : "ring-2 ring-white ring-offset-2 ring-offset-neutral-900 scale-110";
            const pinchedRing = isWhite
              ? "ring-4 ring-neutral-300 scale-125"
              : "ring-4 ring-white scale-125";
            return (
              <button
                key={color.name}
                id={`swatch-${color.name}`}
                onClick={() => setActiveColor(color.value)}
                className={`h-8 w-8 rounded-full transition-all duration-200 max-lg:h-7 max-lg:w-7 max-sm:h-6 max-sm:w-6 ${
                  activeColor === color.value ? selectedRing : "hover:scale-110"
                } ${pinchedColor === color.value ? pinchedRing : ""}`}
                style={{ backgroundColor: color.value }}
                title={color.name}
              />
            );
          })}
        </div>

        {/* Camera Preview */}
        <div
          style={{
            position: "fixed",
            top: `${instructionChrome.cameraTop}px`,
            ...(previewLayout.tier === "mobile"
              ? {
                  left: `${instructionChrome.edge}px`,
                  transform: "none",
                }
              : {
                  left: "50%",
                  transform: "translateX(-50%)",
                }),
            zIndex: 100,
            borderRadius: "8px",
            overflow: "hidden",
            boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
            width: `${previewLayout.w}px`,
            backgroundColor: "#000",
            display: polaroidOpen ? "none" : "block",
          }}
        >
          <div
            style={{
              background: "#e8e8e8",
              height: `${previewLayout.chrome}px`,
              display: "flex",
              alignItems: "center",
              padding: `0 ${Math.round(previewLayout.dot + 3)}px`,
              gap: `${Math.max(4, previewLayout.dot - 2)}px`,
            }}
          >
            <div
              style={{
                width: `${previewLayout.dot}px`,
                height: `${previewLayout.dot}px`,
                borderRadius: "50%",
                background: "#ff5f57",
              }}
            />
            <div
              style={{
                width: `${previewLayout.dot}px`,
                height: `${previewLayout.dot}px`,
                borderRadius: "50%",
                background: "#febc2e",
              }}
            />
            <div
              style={{
                width: `${previewLayout.dot}px`,
                height: `${previewLayout.dot}px`,
                borderRadius: "50%",
                background: "#28c840",
              }}
            />
          </div>
          <div
            style={{
              position: "relative",
              width: "100%",
              height: `${previewLayout.h}px`,
            }}
          >
            <video
              ref={videoRef}
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                display: "block",
                borderRadius: 0,
                transform: "scaleX(-1)",
              }}
              playsInline
              muted
            />
            <canvas
              ref={previewCanvasRef}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: `${previewLayout.w}px`,
                height: `${previewLayout.h}px`,
                pointerEvents: "none",
              }}
            />
          </div>
        </div>

        {/* Loading Overlay */}
        {isModelLoading && (
          <div className="absolute inset-0 z-30 flex flex-col items-center justify-center bg-neutral-950/80 px-4 backdrop-blur-sm">
            <Loader2 className="mb-4 h-12 w-12 animate-spin text-white max-lg:mb-3 max-lg:h-10 max-lg:w-10 max-sm:mb-2.5 max-sm:h-9 max-sm:w-9" />
            <p className="max-w-[min(20rem,88vw)] text-center text-lg font-medium leading-snug text-white max-lg:text-base max-sm:text-sm">
              Loading Hand Tracking Model...
            </p>
            <p className="mt-2 max-w-[min(18rem,85vw)] text-center text-sm leading-snug text-neutral-400 max-lg:mt-1.5 max-lg:text-xs max-sm:mt-1.5 max-sm:text-[11px]">
              Please allow camera access.
            </p>
          </div>
        )}

        {/* Left Panel */}
        <div
          style={{
            position: "absolute",
            bottom: `${instructionChrome.edge}px`,
            left: `${instructionChrome.edge}px`,
            maxWidth:
              previewLayout.tier === "mobile"
                ? "min(44vw, 200px)"
                : previewLayout.tier === "tablet"
                  ? "min(38vw, 220px)"
                  : undefined,
            background: "rgba(255, 255, 255, 0.06)",
            backdropFilter: "blur(8px)",
            border: "0.5px solid rgba(255, 255, 255, 0.12)",
            borderRadius: "8px",
            padding: instructionChrome.pad,
            color: "rgba(255, 255, 255, 0.85)",
            fontFamily: "inherit",
            minWidth: instructionChrome.minW,
            pointerEvents: "none",
            zIndex: 10,
          }}
        >
          <div
            style={{
              fontSize: `${instructionChrome.titleFs}px`,
              fontWeight: 600,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "rgba(255,255,255,0.4)",
              marginBottom: `${instructionChrome.titleMb}px`,
            }}
          >
            Left hand
          </div>
          <div style={{ marginBottom: `${instructionChrome.blockMb}px` }}>
            <div
              style={{
                fontSize: `${instructionChrome.headFs}px`,
                color: "rgba(255,255,255,0.9)",
                fontWeight: 500,
              }}
            >
              Grab & Reposition
            </div>
            <div
              style={{
                fontSize: `${instructionChrome.bodyFs}px`,
                color: "rgba(255,255,255,0.4)",
                fontWeight: 400,
                marginTop: "2px",
                lineHeight: previewLayout.tier === "mobile" ? 1.25 : 1.35,
              }}
            >
              Open hand, then close fist over shape
            </div>
          </div>
          <div style={{ marginBottom: `${instructionChrome.blockMb}px` }}>
            <div
              style={{
                fontSize: `${instructionChrome.headFs}px`,
                color: "rgba(255,255,255,0.9)",
                fontWeight: 500,
              }}
            >
              Resize
            </div>
            <div
              style={{
                fontSize: `${instructionChrome.bodyFs}px`,
                color: "rgba(255,255,255,0.4)",
                fontWeight: 400,
                marginTop: "2px",
                lineHeight: previewLayout.tier === "mobile" ? 1.25 : 1.35,
              }}
            >
              Hold fist + pinch with right hand
            </div>
          </div>
          <div>
            <div
              style={{
                fontSize: `${instructionChrome.headFs}px`,
                color: "rgba(255,255,255,0.9)",
                fontWeight: 500,
              }}
            >
              Release
            </div>
            <div
              style={{
                fontSize: `${instructionChrome.bodyFs}px`,
                color: "rgba(255,255,255,0.4)",
                fontWeight: 400,
                marginTop: "2px",
                lineHeight: previewLayout.tier === "mobile" ? 1.25 : 1.35,
              }}
            >
              Open fist
            </div>
          </div>
        </div>

        {/* Right Panel */}
        <div
          style={{
            position: "absolute",
            bottom: `${instructionChrome.edge}px`,
            right: `${instructionChrome.edge}px`,
            maxWidth:
              previewLayout.tier === "mobile"
                ? "min(44vw, 200px)"
                : previewLayout.tier === "tablet"
                  ? "min(38vw, 220px)"
                  : undefined,
            background: "rgba(255, 255, 255, 0.06)",
            backdropFilter: "blur(8px)",
            border: "0.5px solid rgba(255, 255, 255, 0.12)",
            borderRadius: "8px",
            padding: instructionChrome.pad,
            color: "rgba(255, 255, 255, 0.85)",
            fontFamily: "inherit",
            minWidth: instructionChrome.minW,
            pointerEvents: "none",
            zIndex: 10,
          }}
        >
          <div
            style={{
              fontSize: `${instructionChrome.titleFs}px`,
              fontWeight: 600,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "rgba(255,255,255,0.4)",
              marginBottom: `${instructionChrome.titleMb}px`,
            }}
          >
            Right hand
          </div>
          <div style={{ marginBottom: `${instructionChrome.blockMb}px` }}>
            <div
              style={{
                fontSize: `${instructionChrome.headFs}px`,
                color: "rgba(255,255,255,0.9)",
                fontWeight: 500,
              }}
            >
              Draw
            </div>
            <div
              style={{
                fontSize: `${instructionChrome.bodyFs}px`,
                color: "rgba(255,255,255,0.4)",
                fontWeight: 400,
                marginTop: "2px",
                lineHeight: previewLayout.tier === "mobile" ? 1.25 : 1.35,
              }}
            >
              Point index finger
            </div>
          </div>
          <div style={{ marginBottom: `${instructionChrome.blockMb}px` }}>
            <div
              style={{
                fontSize: `${instructionChrome.headFs}px`,
                color: "rgba(255,255,255,0.9)",
                fontWeight: 500,
              }}
            >
              Complete Drawing
            </div>
            <div
              style={{
                fontSize: `${instructionChrome.bodyFs}px`,
                color: "rgba(255,255,255,0.4)",
                fontWeight: 400,
                marginTop: "2px",
                lineHeight: previewLayout.tier === "mobile" ? 1.25 : 1.35,
              }}
            >
              Hold palm open for {FINISH_PALM_HOLD_MS / 1000}s
            </div>
          </div>
          <div style={{ marginBottom: `${instructionChrome.blockMb}px` }}>
            <div
              style={{
                fontSize: `${instructionChrome.headFs}px`,
                color: "rgba(255,255,255,0.9)",
                fontWeight: 500,
              }}
            >
              Pick Color
            </div>
            <div
              style={{
                fontSize: `${instructionChrome.bodyFs}px`,
                color: "rgba(255,255,255,0.4)",
                fontWeight: 400,
                marginTop: "2px",
                lineHeight: previewLayout.tier === "mobile" ? 1.25 : 1.35,
              }}
            >
              Pinch index + thumb over swatch
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
