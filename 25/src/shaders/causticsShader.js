// Physical Lagrangian Refracted Light Sheet Shaders
// Implements a Phillips/Kolmogorov multi-scale capillary-gravity wave spectrum
// with nonlinear Stokes asymmetry, multi-layer orbital advection, wave-packet intermittency,
// and 3rd-derivative catastrophe optics.
// - Long carrier swells provide large slope (horizontal ray transport) & fast group sweeping.
// - Medium & short ripples carry high curvature (producing intricate pleated sub-folds,
//   sharp triangular swallowtail cusps, and uneven ribbon vs. hair-thin filament widths).

export const causticsVertexShader = /* glsl */ `
  uniform float uTime;
  uniform float uDepth;           // Water depth in meters (0.0 to 2.2m)
  uniform vec2 uCameraPos;        // Current camera center in world meters
  uniform vec2 uMeshSpan;         // World-space width & height of the surface mesh in meters
  uniform float uGridSnap;        // World-space grid step for zero-shimmer snapping
  uniform float uRefractionScale; // Overall refraction strength

  varying float vMu1;             // Principal Jacobian eigenvalue 1 (1 - alpha * D * lambda1)
  varying float vMu2;             // Principal Jacobian eigenvalue 2 (1 - alpha * D * lambda2)
  varying float vDetJ;            // Signed Jacobian determinant (mu1 * mu2)
  varying float vFoldSteepness;   // 3rd-derivative surface steepness |d(mu)/dx0| controlling ray thickness
  varying float vPacketEnergy;    // Local wave-group energy modulation
  varying float vRibbonThick;     // Spatial thickness modulator (thin filaments vs wide trunks & intersections)

  // Nonlinear anharmonic Stokes wave gradient contribution with transverse crest modulation
  vec2 stokesWaveSlope(
    vec2 p,
    vec2 dir,
    float k,
    float slopeAmp,
    float speed,
    float phase,
    float stokesAsym,
    float crestModFreq,
    float crestModSpeed
  ) {
    vec2 ortho = vec2(-dir.y, dir.x);
    float theta = dot(p, dir) * k - uTime * speed + phase;

    // Transverse modulation along the wave crest so ridges naturally pinch, branch, and taper
    float crestPhase = dot(p, ortho) * (k * crestModFreq) - uTime * crestModSpeed + phase * 1.73;
    float crestEnv = 0.67 + 0.33 * sin(crestPhase);

    // Nonlinear Stokes harmonic sharpens focusing troughs/crests asymmetrically
    float dPsi = -sin(theta) - stokesAsym * sin(2.0 * theta);
    float dEnv = 0.33 * crestModFreq * cos(crestPhase);

    // Product rule gradient of modulated wave
    return dir * (slopeAmp * dPsi * crestEnv)
         + ortho * (slopeAmp * (cos(theta) + 0.5 * stokesAsym * cos(2.0 * theta)) * dEnv);
  }

  // Evaluate the physical water surface slope grad(h) at world coordinate p0
  vec2 evalWaterSlope(vec2 p0, out float outPacketEnergy, out float outRibbonThick) {
    // Map full pool depth [0.0m, 2.0m] onto the optical range [0.50m, 0.95m]:
    // - uDepth = 0.0m starts at the 0.50m visual lightrays
    // - uDepth = 2.0m reaches the 0.95m peak visual (never going past peak focus)
    float depthT = clamp(uDepth / 2.0, 0.0, 1.0);

    // Slightly more spread-out base spatial spacing (1.08) so rays have room to breathe
    const float SPATIAL_DENSITY = 1.08;
    vec2 p = p0 * SPATIAL_DENSITY;

    // --- Layer 0: Fast-Moving Long Gravity Swells & Wave-Group Intermittency ---
    // Long swells travel fastest (v_phase = sqrt(g/k)), carrying large horizontal slope
    // that sweeps and advects the entire caustic web unevenly across the pool floor.
    float g1 = sin(dot(p, vec2( 0.866,  0.500)) * 4.8 - uTime * 2.35 + 0.3);
    float g2 = sin(dot(p, vec2(-0.422,  0.906)) * 6.4 - uTime * 2.80 + 1.9);
    float g3 = sin(dot(p, vec2( 0.309, -0.951)) * 4.1 + uTime * 1.65 + 4.1);
    float g4 = sin(dot(p, vec2(-0.920, -0.391)) * 7.8 - uTime * 3.20 + 2.7);

    // Spatially & temporally uneven wave-packet envelopes (some regions calm, others knotted & intense)
    float packetEnv1 = clamp(0.88 + 0.44 * (g1 * 0.55 + g2 * 0.45 + g3 * 0.35), 0.44, 1.68);
    float packetEnv2 = clamp(0.86 + 0.44 * (g2 * 0.50 - g3 * 0.45 + g4 * 0.45), 0.42, 1.62);
    float packetEnv3 = clamp(0.85 + 0.48 * (g4 * 0.55 + g1 * 0.40 - g2 * 0.35), 0.40, 1.65);
    outPacketEnergy = 0.35 * packetEnv1 + 0.35 * packetEnv2 + 0.30 * packetEnv3;

    // Fast macro carrier swell slope & orbital advection vector
    vec2 swellSlope = vec2(0.0);
    swellSlope += vec2( 0.866,  0.500) * (-0.0135 * cos(dot(p, vec2( 0.866,  0.500)) * 6.8 - uTime * 2.95 + 0.5));
    swellSlope += vec2(-0.422,  0.906) * (-0.0112 * cos(dot(p, vec2(-0.422,  0.906)) * 8.5 - uTime * 3.30 + 2.1));
    swellSlope += vec2( 0.258, -0.966) * (-0.0120 * cos(dot(p, vec2( 0.258, -0.966)) * 5.6 - uTime * 2.15 + 3.8));

    // --- Layer 1: Primary Sinuous Wave Arteries (Advected by Macro Swells) ---
    // At 0m (depthT = 0), primary arteries are more spaced out; secondary crossing arteries
    // ramp in across depthT so ray density builds progressively as you dive deeper.
    float primaryDensityRamp = mix(0.45, 1.0, smoothstep(0.0, 0.85, depthT));
    vec2 p1 = p + swellSlope * 2.6;
    vec2 slope1 = vec2(0.0);

    slope1 += stokesWaveSlope(p1, vec2( 0.9511,  0.3090), 13.8, 0.0144, 2.35, 0.40, 0.28, 0.30, 0.50);
    slope1 += stokesWaveSlope(p1, vec2(-0.5878,  0.8090), 17.5, 0.0125, 1.45, 1.85, 0.25, 0.35, 0.62);
    slope1 += stokesWaveSlope(p1, vec2(-0.8910, -0.4540), 12.2, 0.0154, 2.55, 3.10, 0.30, 0.26, 0.42);
    slope1 += stokesWaveSlope(p1, vec2( 0.4067, -0.9135), 20.8, 0.0110 * primaryDensityRamp, 1.05, 4.70, 0.24, 0.38, 0.72);
    slope1 += stokesWaveSlope(p1, vec2( 0.1045,  0.9945), 15.6, 0.0131, 1.85, 2.30, 0.26, 0.32, 0.55);
    slope1 += stokesWaveSlope(p1, vec2( 0.7880, -0.6157), 18.8, 0.0116 * primaryDensityRamp, 1.65, 5.20, 0.25, 0.34, 0.58);

    slope1 *= packetEnv1;

    // --- Layer 2: Medium Crossing Wave Trains (Advected by Layer 0 + Layer 1) ---
    // Starts at the 0.50m level (0.46) at 0m depth and ramps smoothly to 1.0 at 2.0m depth,
    // gradually weaving more crossing rays and junctions as you swim deeper.
    float layer2DepthScale = mix(0.46, 1.0, smoothstep(0.0, 1.0, depthT));
    vec2 p2 = p1 + slope1 * (1.15 + 0.60 * layer2DepthScale);
    vec2 slope2 = vec2(0.0);

    slope2 += stokesWaveSlope(p2, vec2( 0.7431,  0.6691), 26.5, 0.0085, 1.15, 0.95, 0.24, 0.38, 0.75);
    slope2 += stokesWaveSlope(p2, vec2(-0.7193,  0.6947), 32.8, 0.0074, 2.10, 2.65, 0.22, 0.42, 0.92);
    slope2 += stokesWaveSlope(p2, vec2(-0.9781,  0.2079), 23.8, 0.0090, 0.92, 4.15, 0.25, 0.34, 0.68);
    slope2 += stokesWaveSlope(p2, vec2( 0.6157, -0.7880), 37.5, 0.0065 * depthT, 2.45, 1.20, 0.20, 0.45, 1.10);
    slope2 += stokesWaveSlope(p2, vec2(-0.2079, -0.9781), 29.8, 0.0078, 1.58, 5.40, 0.22, 0.40, 0.85);
    slope2 += stokesWaveSlope(p2, vec2( 0.8829, -0.4695), 42.5, 0.0058 * depthT, 2.85, 3.75, 0.18, 0.48, 1.25);
    slope2 += stokesWaveSlope(p2, vec2(-0.4695, -0.8829), 34.6, 0.0068 * depthT, 1.92, 2.15, 0.21, 0.43, 0.98);

    slope2 *= packetEnv2 * layer2DepthScale;

    // --- Layer 3: Fast Capillary & Pleated Sub-Ribbon Ripples (Advected by Layer 2) ---
    // Gradually introduces fine sub-filaments across the lower two-thirds of the dive toward the peak.
    float layer3DepthScale = smoothstep(0.18, 1.0, depthT) * 0.88;
    vec2 p3 = p2 + slope2 * 1.30;
    float localSteepness = clamp(0.48 + length(slope1 + slope2) * 32.0, 0.45, 1.75) * packetEnv3 * layer3DepthScale;
    vec2 slope3 = vec2(0.0);

    slope3 += stokesWaveSlope(p3, vec2( 0.9272,  0.3746), 52.0, 0.0036, 2.95, 0.60, 0.16, 0.50, 1.45);
    slope3 += stokesWaveSlope(p3, vec2(-0.3746,  0.9272), 61.5, 0.0031, 3.60, 2.10, 0.15, 0.52, 1.70);
    slope3 += stokesWaveSlope(p3, vec2(-0.7660, -0.6428), 56.5, 0.0034, 3.25, 4.40, 0.16, 0.48, 1.55);
    slope3 += stokesWaveSlope(p3, vec2( 0.5299, -0.8480), 69.0, 0.0027, 4.10, 1.55, 0.14, 0.55, 1.95);
    slope3 += stokesWaveSlope(p3, vec2( 0.1736,  0.9848), 78.0, 0.0023, 4.55, 3.30, 0.12, 0.58, 2.20);
    slope3 += stokesWaveSlope(p3, vec2(-0.8480,  0.5299), 86.0, 0.0019, 4.95, 5.15, 0.12, 0.60, 2.40);

    slope3 *= localSteepness;

    // Spatial thickness modulator: high where macro swells & primary trunks dominate,
    // low where capillary ripples dominate—creating dramatic variety from hair-thin threads to wide ribbons.
    float macroDominance = length(swellSlope + slope1) / (length(slope3) * 1.6 + 0.006);
    float groupThickWave = 0.5 + 0.5 * sin(dot(p, vec2(0.62, -0.78)) * 5.2 - uTime * 1.4 + g1 * 1.2)
                               * cos(dot(p, vec2(0.81,  0.58)) * 4.5 + uTime * 1.1 + g2 * 1.2);
    outRibbonThick = clamp(
      (0.35 + 0.65 * smoothstep(0.65, 2.1, macroDominance)) * (0.45 + 1.55 * pow(groupThickWave, 0.75)) * packetEnv1,
      0.22,
      2.40
    );

    return (swellSlope + slope1 + slope2 + slope3) / SPATIAL_DENSITY;
  }

  // Refracted ray intersection on the pool floor at water depth uDepth
  vec2 computeFloorPos(vec2 x0, float alphaD, out float packetEnergy, out float ribbonThick) {
    vec2 slope = evalWaterSlope(x0, packetEnergy, ribbonThick);
    return x0 - alphaD * slope;
  }

  void main() {
    // Snap mesh center to world grid step to prevent vertex swimming while moving
    vec2 snappedCenter = floor(uCameraPos / uGridSnap) * uGridSnap;
    vec2 x0 = snappedCenter + position.xy * uMeshSpan;

    // Map full pool depth uDepth in [0.0m, 2.0m] onto effective optical depth effDepth in [0.50m, 0.95m]:
    // - At uDepth = 0.0m (top): matches the 0.50m visual lightrays (alphaD ~ 0.45)
    // - At uDepth = 2.0m (bottom): reaches the 0.95m peak visual (alphaD = 1.786), never going past peak!
    float depthT = clamp(uDepth / 2.0, 0.0, 1.0);
    float effDepth = mix(0.50, 0.95, depthT);
    float normD = effDepth / 0.95;
    float focusCurve = 0.035 * normD + 0.965 * pow(normD, 2.25);
    float alphaD = (uRefractionScale * 0.95) * focusCurve;

    // Evaluate refracted floor position at vertex and symmetric stencil neighbors
    // to obtain BOTH the exact 2x2 Jacobian J AND the 3rd-derivative fold steepness!
    float eps = 0.0018;
    float energyC, ribbonThickC, dummy1, dummy2;

    vec2 posC  = computeFloorPos(x0, alphaD, energyC, ribbonThickC);
    vec2 posXp = computeFloorPos(x0 + vec2(eps, 0.0), alphaD, dummy1, dummy2);
    vec2 posXm = computeFloorPos(x0 - vec2(eps, 0.0), alphaD, dummy1, dummy2);
    vec2 posYp = computeFloorPos(x0 + vec2(0.0, eps), alphaD, dummy1, dummy2);
    vec2 posYm = computeFloorPos(x0 - vec2(0.0, eps), alphaD, dummy1, dummy2);

    // Central-difference exact Jacobian columns Jx = d(floorPos)/dx0, Jy = d(floorPos)/dy0
    vec2 Jx = (posXp - posXm) / (2.0 * eps);
    vec2 Jy = (posYp - posYm) / (2.0 * eps);

    // Second derivative of floorPos (= -alpha * D * third derivative of water surface h'''!)
    // High value -> razor-thin filament; Low value -> wide, thick focal ribbon!
    vec2 d2x = (posXp - 2.0 * posC + posXm) / (eps * eps);
    vec2 d2y = (posYp - 2.0 * posC + posYm) / (eps * eps);
    float curvatureGrad = (sqrt(dot(d2x, d2x) + dot(d2y, d2y)) / max(0.18, alphaD * 0.53)) / 1.08;

    // Extract principal stretch eigenvalues (mu1, mu2) of the 2x2 Jacobian matrix J
    float traceJ = Jx.x + Jy.y;
    float detJ = Jx.x * Jy.y - Jx.y * Jy.x;
    float diffJ = 0.5 * (Jx.x - Jy.y);
    float shearJ = 0.5 * (Jx.y + Jy.x);
    float disc = sqrt(max(0.0, diffJ * diffJ + shearJ * shearJ));

    // Ordered so mu1 is the most strongly focused (minimum / most negative) eigenvalue
    float mu1 = 0.5 * traceJ - disc;
    float mu2 = 0.5 * traceJ + disc;

    vMu1 = mu1;
    vMu2 = mu2;
    vDetJ = detJ;
    vFoldSteepness = clamp(curvatureGrad, 4.0, 95.0);
    vPacketEnergy = energyC;
    vRibbonThick = ribbonThickC;

    gl_Position = projectionMatrix * viewMatrix * vec4(posC, 0.0, 1.0);
  }
`;

export const causticsFragmentShader = /* glsl */ `
  precision highp float;

  uniform float uDepth;

  varying float vMu1;
  varying float vMu2;
  varying float vDetJ;
  varying float vFoldSteepness;
  varying float vPacketEnergy;
  varying float vRibbonThick;

  void main() {
    // Map full pool depth uDepth in [0.0m, 2.0m] onto effective optical depth effDepth in [0.50m, 0.95m]
    float depthT = clamp(uDepth / 2.0, 0.0, 1.0);
    float effDepth = mix(0.50, 0.95, depthT);

    // --- 1. Physical Luminous Flux Conservation & Focus Progression ---
    // Combine 3rd-derivative steepness with spatial vRibbonThick so capillary threads stay
    // hair-thin (steepNorm ~ 3.2) while primary trunks & intersections open up wide (steepNorm ~ 0.18).
    float steepNorm = clamp((vFoldSteepness / 34.0) / max(0.30, vRibbonThick * 0.85), 0.18, 3.4);

    // Detect proximity to a line intersection (where the transverse eigenvalue vMu2 also converges toward 0)
    float intersectProximity1 = smoothstep(0.34, -0.08, vMu2);
    float intersectProximity2 = smoothstep(0.34, -0.08, vMu1);

    // Where lines intersect, the ray thickness flares outward significantly (especially on high-vRibbonThick trunks)
    float flare1 = 1.0 + intersectProximity1 * (0.55 + 1.05 * vRibbonThick);
    float flare2 = 1.0 + intersectProximity2 * (0.55 + 1.05 * vRibbonThick);

    // At 0m (effDepth = 0.50m), shallowBlur is ~0.20 (soft glowing rays); at 2.0m (effDepth = 0.95m), shallowBlur is 0.0 (peak sharp)
    float shallowBlur = mix(0.42, 0.0, smoothstep(0.14, 0.82, effDepth));
    float eps1 = shallowBlur + (0.13 * sqrt(steepNorm)) / clamp(flare1 * 0.65, 0.65, 1.55);
    float eps2 = shallowBlur + 0.15 / clamp(flare2 * 0.65, 0.65, 1.55);

    float invStretch1 = 1.0 / sqrt(vMu1 * vMu1 + eps1 * eps1);
    float invStretch2 = 1.0 / sqrt(vMu2 * vMu2 + eps2 * eps2);

    // Transverse divergence attenuation:
    // At shallow depths (depthT near 0), attenuate secondary diverging branches slightly more so rays look
    // more spaced out, then fill in the full dense web as depthT approaches 1.0 (bottom peak).
    float divThreshold = mix(0.96, 1.06, depthT);
    float divAttenRate = mix(0.75, 0.38, depthT);
    float divExcess = max(0.0, vMu2 - divThreshold);
    float transverseAtten = 1.0 / (1.0 + divAttenRate * divExcess * divExcess);

    float sheetIrradiance = invStretch1 * invStretch2 * transverseAtten;

    // --- 2. Variable-Thickness Caustic Fold, Asymmetric Ribbon Body & Intersection Flare ---
    // Smoothly transitions from the 0.50m soft-ray look at uDepth = 0.0m to the 0.95m razor-sharp peak at uDepth = 2.0m
    float depthFocusFactor = smoothstep(0.35, 0.88, effDepth);
    float unfocusSpread = pow(clamp((0.88 - effDepth) / 0.74, 0.0, 1.0), 1.55) * 0.38;

    float baseSigma1 = clamp(0.086 / pow(steepNorm, 0.70), 0.018, 0.25) + unfocusSpread;
    float baseSigma2 = clamp(0.074 / pow(steepNorm, 0.60), 0.018, 0.21) + unfocusSpread;

    // Asymmetric fold profile:
    // - On the outside of a fold (mu > 0), keep a crisp caustic edge while flaring smoothly at intersections.
    // - On the inside of a folded ribbon (mu < 0) and across intersections, expand sigma wider
    //   so thick rays and intersection hubs are lush, filled luminous bands rather than hollow thin loops!
    float sigmaOuter1 = baseSigma1 * (1.0 + 0.48 * intersectProximity1 * vRibbonThick);
    float sigmaInner1 = min(0.50, baseSigma1 * (1.35 + 0.95 * vRibbonThick) * flare1);
    float sigma1 = (vMu1 >= 0.0) ? sigmaOuter1 : sigmaInner1;

    float sigmaOuter2 = baseSigma2 * (1.0 + 0.48 * intersectProximity2 * vRibbonThick);
    float sigmaInner2 = min(0.45, baseSigma2 * (1.35 + 0.95 * vRibbonThick) * flare2);
    float sigma2 = (vMu2 >= 0.0) ? sigmaOuter2 : sigmaInner2;

    float u1 = vMu1 / sigma1;
    float u2 = vMu2 / sigma2;

    float coreProfile1 = exp(-0.48 * u1 * u1) / (1.0 + 0.28 * u1 * u1);
    float coreProfile2 = exp(-0.48 * u2 * u2) / (1.0 + 0.28 * u2 * u2);

    // Weight by transverse convergence: secondary filaments progressively brighten with depthT
    float densityScale = mix(0.70, 0.82, depthT);
    float branchWeight1 = pow(clamp(invStretch2 * densityScale, 0.10, 3.6), 1.32) * transverseAtten;
    float branchWeight2 = pow(clamp(invStretch1 * densityScale, 0.10, 3.6), 1.32);

    // Hyperbolic Intersection Webbing:
    // Where two rays cross (both |mu1| and |mu2| small), light forms a wide, curved hyperbolic bridge/fillet
    // that thickens the intersection hub depending on local vRibbonThick.
    float webRadius = clamp(0.09 + 0.12 * vRibbonThick, 0.09, 0.30);
    float webRadialEnv = exp(-0.55 * (vMu1 * vMu1 + vMu2 * vMu2) / (webRadius * webRadius));
    float hypScale = 0.014 + 0.036 * vRibbonThick;
    float intersectionWeb = exp(-abs(vDetJ) / hypScale) * webRadialEnv * (0.65 + 0.55 * vRibbonThick);

    float crispEdge = (coreProfile1 * branchWeight1 + coreProfile2 * branchWeight2 + intersectionWeb * 1.25)
                      * depthFocusFactor
                      * (0.65 + 0.45 * vPacketEnergy);

    // Boost sheetIrradiance inside thick ribbons and flared intersections when focused
    sheetIrradiance += (coreProfile1 * branchWeight1 * 0.30 + intersectionWeb * 0.95)
                       * depthFocusFactor
                       * max(0.0, vRibbonThick - 0.65);

    // --- 3. Umbilic & Swallowtail Star Knots (where both mu1 and mu2 converge) ---
    // Scale intersection hotspot size with vRibbonThick so major line intersections form broad, thick luminous hubs
    float cuspSigma = clamp(0.068 + 0.072 * vRibbonThick, 0.065, 0.235);
    float c1 = exp(-0.48 * (vMu1 * vMu1) / (cuspSigma * cuspSigma));
    float c2 = exp(-0.48 * (vMu2 * vMu2) / (cuspSigma * cuspSigma));
    float cuspHotspot = (pow(c1 * c2, 1.05) * 2.5 + intersectionWeb * 0.75)
                        * smoothstep(0.48, 0.88, effDepth)
                        * vPacketEnergy;

    // Graceful high-dynamic-range clamping
    sheetIrradiance = min(sheetIrradiance, 7.0);
    crispEdge = min(crispEdge, 6.5);
    cuspHotspot = min(cuspHotspot, 7.0);

    gl_FragColor = vec4(sheetIrradiance, crispEdge, cuspHotspot, 1.0);
  }
`;
