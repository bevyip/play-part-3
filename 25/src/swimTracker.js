// MediaPipe Hand Tracking for Breaststroke Swimming Control
// Detects two hands coming together in the center ("catch/streamline") and pushing
// outwards with palms facing out ("power stroke") to dive deeper and glide forward.
// When the swimming motion stops, buoyancy smoothly floats the depth back to shallow water.

const MEDIAPIPE_VISION_CDN =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
const MEDIAPIPE_WASM_ROOT =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

// Hand skeleton bone connections for visual overlay
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [0, 13], [13, 14], [14, 15], [15, 16],// Ring
  [0, 17], [17, 18], [18, 19], [19, 20],// Pinky
  [5, 9], [9, 13], [13, 17], [0, 17],   // Palm arch
];

export const SHALLOW_FLOAT_DEPTH = 0.00; // 0 m starting depth at surface (maps to 0.50m visual lightrays)
export const MAX_SWIM_DEPTH = 2.00;      // 2.00 m bottom of pool (maps to 0.95m peak visual)

export function createSwimTracker(state) {
  const panelEl = document.getElementById('swim-cam-panel');
  const bodyEl = document.getElementById('swim-cam-body');
  const videoEl = document.getElementById('swim-cam-video');
  const overlayCanvasEl = document.getElementById('swim-cam-canvas');
  const hintEl = document.getElementById('swim-hint');
  const frameHelperEl = document.getElementById('swim-cam-helper');
  const ctx = overlayCanvasEl ? overlayCanvasEl.getContext('2d') : null;

  // One quiet window per warning. A later exit inside this span does not show it again.
  const frameWarnSec = frameHelperEl
    ? (parseFloat(getComputedStyle(frameHelperEl).getPropertyValue('--frame-warn-duration')) || 3)
    : 3;
  let everHadBothHands = false;
  let handsWereInFrame = false;
  let handsOutFor = 0;
  let frameWarnQuietUntil = -1;

  if (frameHelperEl) {
    frameHelperEl.addEventListener('animationend', () => {
      frameHelperEl.classList.remove('is-visible');
      frameHelperEl.setAttribute('aria-hidden', 'true');
    });
  }

  function showFrameWarning(nowSec) {
    if (!frameHelperEl) return;
    frameHelperEl.classList.remove('is-visible');
    void frameHelperEl.offsetWidth;
    frameHelperEl.classList.add('is-visible');
    frameHelperEl.setAttribute('aria-hidden', 'false');
    frameWarnQuietUntil = nowSec + frameWarnSec;
  }

  function updateFrameWarning(handCount, nowSec, frameDt) {
    if (handCount >= 2) {
      everHadBothHands = true;
      handsWereInFrame = true;
      handsOutFor = 0;
      return;
    }
    if (!everHadBothHands || !handsWereInFrame) return;

    handsOutFor += frameDt;
    if (handsOutFor < 0.28) return;

    handsWereInFrame = false;
    handsOutFor = 0;
    if (nowSec < frameWarnQuietUntil) return;
    showFrameWarning(nowSec);
  }

  let isEnabled = false;
  let isLoading = false;
  let handLandmarker = null;
  let mediaStream = null;
  let lastVideoTime = -1;

  // Smoothed hand tracking & stroke state
  const tracker = {
    hasTwoHands: false,
    smoothedSepX: 0,
    smoothedDist: 0,
    prevSepX: null,
    prevLeftX: null,
    prevRightX: null,
    leftPalm: { x: 0.35, y: 0.5 },
    rightPalm: { x: 0.65, y: 0.5 },
    strokeArmed: false,       // True once hands have come together in the center catch zone
    catchMinSep: 1.0,         // Minimum separation reached during current catch phase
    isPushingOut: false,      // True during active outward power sweep
    isRecovering: false,      // True while bringing hands back together after a stroke
    lastStrokeTime: -10.0,    // Timestamp (sec) of last active outward push
    lastRecoveryTime: -10.0,  // Timestamp (sec) of active inward recovery motion
    manualHoldUntil: -10.0,   // Grace period if user manually clicks a preset or scrolls
    strokeIntensity: 0.0,     // 0..1 smoothed visual indicator of current outward push
    palmOutScore: 1.0,
  };

  async function initMediaPipe() {
    if (handLandmarker) return handLandmarker;
    const visionModule = await import(/* @vite-ignore */ MEDIAPIPE_VISION_CDN);
    const { FilesetResolver, HandLandmarker } = visionModule;
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_ROOT);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: HAND_MODEL_URL,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.45,
      minHandPresenceConfidence: 0.45,
      minTrackingConfidence: 0.45,
    });
    return handLandmarker;
  }

  async function startTracking() {
    if (isLoading || isEnabled) return;
    isLoading = true;

    try {
      const [landmarker, stream] = await Promise.all([
        initMediaPipe(),
        navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 320 },
            height: { ideal: 240 },
            facingMode: 'user',
          },
          audio: false,
        }),
      ]);

      mediaStream = stream;
      videoEl.srcObject = stream;
      await videoEl.play();

      overlayCanvasEl.width = videoEl.videoWidth || 320;
      overlayCanvasEl.height = videoEl.videoHeight || 240;

      isEnabled = true;
      isLoading = false;
      lastVideoTime = -1;
      tracker.prevSepX = null;
      tracker.strokeArmed = false;
      tracker.lastStrokeTime = state.time;

      if (panelEl) panelEl.classList.add('active');
      if (bodyEl) bodyEl.hidden = false;
    } catch (err) {
      console.error('Failed to start hand swim tracking:', err);
      isLoading = false;
      if (panelEl) {
        panelEl.addEventListener('click', () => startTracking(), { once: true });
      }
    }
  }

  // Automatically enable webcam hand tracking on render
  startTracking();

  // Compute palm center and outward palm orientation in mirrored normalized coords (mx = 1 - x)
  function analyzeHand(landmarks) {
    const wrist = { x: 1.0 - landmarks[0].x, y: landmarks[0].y };
    const thumbTip = { x: 1.0 - landmarks[4].x, y: landmarks[4].y };
    const indexMcp = { x: 1.0 - landmarks[5].x, y: landmarks[5].y };
    const middleMcp = { x: 1.0 - landmarks[9].x, y: landmarks[9].y };
    const middleTip = { x: 1.0 - landmarks[12].x, y: landmarks[12].y };
    const pinkyMcp = { x: 1.0 - landmarks[17].x, y: landmarks[17].y };
    const pinkyTip = { x: 1.0 - landmarks[20].x, y: landmarks[20].y };

    const palm = {
      x: (wrist.x + indexMcp.x + middleMcp.x + pinkyMcp.x) * 0.25,
      y: (wrist.y + indexMcp.y + middleMcp.y + pinkyMcp.y) * 0.25,
    };

    return {
      landmarks,
      wrist,
      thumbTip,
      indexMcp,
      middleMcp,
      middleTip,
      pinkyMcp,
      pinkyTip,
      palm,
    };
  }

  function drawOverlay(hands, nowSec) {
    if (!ctx || !overlayCanvasEl) return;
    const w = overlayCanvasEl.width;
    const h = overlayCanvasEl.height;
    ctx.clearRect(0, 0, w, h);

    if (hands.length === 0) return;

    // 1. If 2 hands detected, draw connection bar & outward propulsion arrows
    if (hands.length === 2) {
      const lx = tracker.leftPalm.x * w;
      const ly = tracker.leftPalm.y * h;
      const rx = tracker.rightPalm.x * w;
      const ry = tracker.rightPalm.y * h;

      ctx.save();
      ctx.strokeStyle = tracker.isPushingOut
        ? 'rgba(155, 245, 255, 0.92)'
        : tracker.strokeArmed
          ? 'rgba(120, 225, 255, 0.65)'
          : 'rgba(140, 200, 230, 0.35)';
      ctx.lineWidth = tracker.isPushingOut ? 3.0 : 1.6;
      ctx.beginPath();
      ctx.moveTo(lx, ly);
      ctx.lineTo(rx, ry);
      ctx.stroke();

      // Draw outward push arrows when stroke is armed or actively pushing out
      if (tracker.isPushingOut || tracker.strokeArmed) {
        const arrowLen = tracker.isPushingOut ? 22 : 12;
        ctx.fillStyle = tracker.isPushingOut ? '#b8f7ff' : 'rgba(165, 235, 255, 0.75)';
        // Left outward arrow
        ctx.beginPath();
        ctx.moveTo(lx - arrowLen, ly);
        ctx.lineTo(lx - arrowLen + 7, ly - 5);
        ctx.lineTo(lx - arrowLen + 7, ly + 5);
        ctx.closePath();
        ctx.fill();
        // Right outward arrow
        ctx.beginPath();
        ctx.moveTo(rx + arrowLen, ry);
        ctx.lineTo(rx + arrowLen - 7, ry - 5);
        ctx.lineTo(rx + arrowLen - 7, ry + 5);
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();
    }

    // 3. Draw each hand skeleton in mirrored coordinates
    hands.forEach((hand) => {
      const lm = hand.landmarks;
      ctx.save();
      ctx.strokeStyle = tracker.isPushingOut
        ? 'rgba(165, 248, 255, 0.92)'
        : 'rgba(110, 212, 248, 0.78)';
      ctx.lineWidth = 1.8;

      for (const [i, j] of HAND_CONNECTIONS) {
        const x1 = (1.0 - lm[i].x) * w;
        const y1 = lm[i].y * h;
        const x2 = (1.0 - lm[j].x) * w;
        const y2 = lm[j].y * h;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }

      // Draw joints
      ctx.fillStyle = '#ebfcff';
      for (let i = 0; i < lm.length; i++) {
        const x = (1.0 - lm[i].x) * w;
        const y = lm[i].y * h;
        ctx.beginPath();
        ctx.arc(x, y, i === 0 || i === 9 ? 3.2 : 2.1, 0, Math.PI * 2);
        ctx.fill();
      }

      // Highlight palm center
      ctx.fillStyle = tracker.isPushingOut ? '#ffffff' : '#7ce0ff';
      ctx.beginPath();
      ctx.arc(hand.palm.x * w, hand.palm.y * h, 5.0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });
  }

  function update(dt, nowSec) {
    const isFloatUpLocked = tracker.floatUpLockUntil && nowSec < tracker.floatUpLockUntil;

    if (isEnabled && handLandmarker && videoEl && videoEl.readyState >= 2) {
      // Run MediaPipe inference whenever the webcam has a new video frame
      if (videoEl.currentTime !== lastVideoTime) {
        const videoDt =
          lastVideoTime >= 0 ? Math.max(0.012, Math.min(0.1, videoEl.currentTime - lastVideoTime)) : dt;
        lastVideoTime = videoEl.currentTime;

        const results = handLandmarker.detectForVideo(videoEl, performance.now());
        const rawHands = (results && results.landmarks) ? results.landmarks : [];
        const analyzed = rawHands.map(analyzeHand);

        if (analyzed.length >= 2) {
          // Sort by mirrored X so leftHand is on the left side of screen, rightHand on the right
          analyzed.sort((a, b) => a.palm.x - b.palm.x);
          const leftH = analyzed[0];
          const rightH = analyzed[1];

          const rawSepX = Math.max(0.0, rightH.palm.x - leftH.palm.x);
          const rawDist = Math.hypot(rawSepX, rightH.palm.y - leftH.palm.y);

          const smooth = 0.45;
          if (tracker.prevSepX === null) {
            tracker.smoothedSepX = rawSepX;
            tracker.smoothedDist = rawDist;
            tracker.leftPalm.x = leftH.palm.x;
            tracker.leftPalm.y = leftH.palm.y;
            tracker.rightPalm.x = rightH.palm.x;
            tracker.rightPalm.y = rightH.palm.y;
          } else {
            tracker.smoothedSepX += (rawSepX - tracker.smoothedSepX) * smooth;
            tracker.smoothedDist += (rawDist - tracker.smoothedDist) * smooth;
            tracker.leftPalm.x += (leftH.palm.x - tracker.leftPalm.x) * smooth;
            tracker.leftPalm.y += (leftH.palm.y - tracker.leftPalm.y) * smooth;
            tracker.rightPalm.x += (rightH.palm.x - tracker.rightPalm.x) * smooth;
            tracker.rightPalm.y += (rightH.palm.y - tracker.rightPalm.y) * smooth;
          }

          const velSep =
            tracker.prevSepX !== null ? (tracker.smoothedSepX - tracker.prevSepX) / videoDt : 0.0;
          const leftVelX =
            tracker.prevLeftX !== null ? (tracker.leftPalm.x - tracker.prevLeftX) / videoDt : 0.0;
          const rightVelX =
            tracker.prevRightX !== null ? (tracker.rightPalm.x - tracker.prevRightX) / videoDt : 0.0;

          tracker.prevSepX = tracker.smoothedSepX;
          tracker.prevLeftX = tracker.leftPalm.x;
          tracker.prevRightX = tracker.rightPalm.x;
          tracker.hasTwoHands = true;

          // Evaluate palm-outward orientation (fingers sweeping outward or pinky on outer edge of palm)
          const leftOutFinger = Math.max(0, (leftH.wrist.x - leftH.middleTip.x) * 3.2);
          const rightOutFinger = Math.max(0, (rightH.middleTip.x - rightH.wrist.x) * 3.2);
          const leftPinkyOut = Math.max(0, (leftH.indexMcp.x - leftH.pinkyMcp.x) * 4.8);
          const rightPinkyOut = Math.max(0, (rightH.pinkyMcp.x - rightH.indexMcp.x) * 4.8);

          const leftOutScore = Math.min(1.0, Math.max(leftOutFinger, leftPinkyOut));
          const rightOutScore = Math.min(1.0, Math.max(rightOutFinger, rightPinkyOut));
          tracker.palmOutScore = 0.85 + 0.45 * (0.5 * (leftOutScore + rightOutScore));

          // 1. Arm stroke when hands come together near the center ("Catch / Streamline")
          if (tracker.smoothedDist < 0.38 || tracker.smoothedSepX < 0.34) {
            tracker.strokeArmed = true;
            tracker.catchMinSep = Math.min(tracker.catchMinSep, tracker.smoothedSepX);
          } else if (velSep < -0.18 && tracker.smoothedSepX < 0.50) {
            // Also arm if user is actively bringing hands inward and turns around inside 0.50
            tracker.strokeArmed = true;
            tracker.catchMinSep = Math.min(tracker.catchMinSep, tracker.smoothedSepX);
          }

          // 2. Detect outward breaststroke power push:
          // Hands were gathered in center (strokeArmed) and are now sweeping outward apart
          const bothMovingOut = leftVelX < -0.02 && rightVelX > 0.02 && velSep > 0.26;
          if (!isFloatUpLocked && tracker.strokeArmed && bothMovingOut && tracker.smoothedSepX < 0.88) {
            tracker.isPushingOut = true;
            tracker.isRecovering = false;
            tracker.lastStrokeTime = nowSec;
            if (hintEl) hintEl.classList.add('dismissed');
            tracker.strokeIntensity = Math.min(1.0, velSep * 0.9);

            // Progressive depth resistance so reaching the bottom (2.0m peak visual) takes ~5-6 deliberate strokes
            const depthRatio = Math.min(1.0, state.targetDepth / MAX_SWIM_DEPTH);
            const depthResistance = 1.0 - 0.36 * depthRatio;
            const strokeDelta = velSep * videoDt * 0.76 * tracker.palmOutScore * depthResistance;
            state.targetDepth = Math.min(MAX_SWIM_DEPTH, state.targetDepth + strokeDelta);

            // Disarm once hands reach full wide extension
            if (tracker.smoothedSepX > 0.78) {
              tracker.strokeArmed = false;
              tracker.catchMinSep = 1.0;
            }
          } else {
            tracker.isPushingOut = false;
            tracker.strokeIntensity *= 0.82;

            // Brief recovery grace while actively bringing hands back together right after a stroke
            if (!isFloatUpLocked && velSep < -0.22 && nowSec - tracker.lastStrokeTime < 1.35) {
              tracker.isRecovering = true;
              tracker.lastRecoveryTime = nowSec;
            } else {
              tracker.isRecovering = false;
            }

            if (tracker.smoothedSepX > 0.76) {
              tracker.strokeArmed = false;
              tracker.catchMinSep = 1.0;
            }
          }
        } else {
          // Fewer than 2 hands visible
          tracker.hasTwoHands = false;
          tracker.isPushingOut = false;
          tracker.isRecovering = false;
          tracker.prevSepX = null;
          tracker.prevLeftX = null;
          tracker.prevRightX = null;
          tracker.strokeIntensity *= 0.8;
        }

        updateFrameWarning(analyzed.length, nowSec, videoDt);
        drawOverlay(analyzed, nowSec);
      }
    }

    // --- Buoyancy: Calmly & Steadily Float Back Up to 0m When Action Stops ---
    const timeSinceStroke = nowSec - Math.max(tracker.lastStrokeTime, tracker.manualHoldUntil);
    const isRecoveringBetweenStrokes =
      tracker.isRecovering && (nowSec - tracker.lastRecoveryTime < 0.45) && timeSinceStroke < 1.35;

    if (!tracker.isPushingOut && !isRecoveringBetweenStrokes && timeSinceStroke > 0.60) {
      // Smoothly ease into a steady, constant upward buoyancy speed
      const floatElapsed = timeSinceStroke - 0.60;
      const buoyancyRamp = Math.min(1.0, floatElapsed / 0.80);
      const smoothRamp = buoyancyRamp * buoyancyRamp * (3.0 - 2.0 * buoyancyRamp);
      const depthAboveShallow = Math.max(0.0, state.targetDepth - SHALLOW_FLOAT_DEPTH);

      if (depthAboveShallow > 0.001) {
        const steadyFloatSpeed = 0.28 * smoothRamp;
        state.targetDepth = Math.max(SHALLOW_FLOAT_DEPTH, state.targetDepth - steadyFloatSpeed * dt);
      }
    }
  }

  function notifyManualDepthChange(nowSec) {
    tracker.manualHoldUntil = nowSec + 2.2;
  }

  function triggerFloatUp(nowSec) {
    tracker.isPushingOut = false;
    tracker.isRecovering = false;
    tracker.strokeArmed = false;
    tracker.lastStrokeTime = nowSec - 1.5;
    tracker.manualHoldUntil = nowSec - 1.5;
    tracker.floatUpLockUntil = nowSec + 0.55;
  }

  return {
    update,
    notifyManualDepthChange,
    triggerFloatUp,
  };
}
