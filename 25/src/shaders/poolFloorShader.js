// Pool Floor Composite Shader
// Renders the infinite mosaic ceramic tile floor (light-blue main zone + dark-blue lane stripes
// with authentic per-tile shade variations and grout lines) illuminated by the physical
// Lagrangian caustic light buffer and underwater scattering bloom.

export const poolFloorVertexShader = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

export const poolFloorFragmentShader = /* glsl */ `
  precision highp float;

  uniform sampler2D uCausticsTex;
  uniform sampler2D uBloomTex;
  uniform vec2 uCausticsTexel;      // 1.0 / causticsRT dimensions for silky sub-pixel AA
  uniform vec2 uCameraPos;          // World center in meters
  uniform vec2 uViewSize;           // Base visible floor width & height in meters
  uniform float uDepth;             // Water depth in meters
  uniform float uTileDistanceScale; // Buoyant Z-axis camera-to-tile distance scale (in & out bob only)

  varying vec2 vUv;

  // High-quality 2D hash for deterministic per-tile blue shade variations
  vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
  }

  float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
  }

  void main() {
    // Pure Z-axis camera distance to the tiled floor (including buoyant in-and-out bobbing)
    vec2 effectiveViewSize = uViewSize * uTileDistanceScale;

    // Map screen UV [0, 1] to world coordinates in meters
    vec2 worldPos = uCameraPos + (vUv - 0.5) * effectiveViewSize;

    // 15 mosaic tiles per meter (matches exact ~15 tiles across portrait screen in screenshots)
    float tilesPerMeter = 15.0;
    vec2 tileCoord = worldPos * tilesPerMeter;
    vec2 tileId = floor(tileCoord);
    vec2 tileFract = fract(tileCoord);

    // Anti-aliased derivative in tile space
    vec2 fw = fwidth(tileCoord);
    float edgeAA = max(max(fw.x, fw.y), 0.004);

    // Determine whether this tile is in the Dark Navy Lane Stripe or Main Light-Blue Pool Zone.
    // At cameraPos = (0, 0), left ~62% of portrait screen (tileId.x < 2) is light-blue tiles,
    // right ~38% (tileId.x >= 2) is dark-blue lane stripe tiles, repeating every 26 tiles as you swim.
    float lanePeriod = 26.0;
    float laneIndex = mod(tileId.x + 16.0, lanePeriod);
    bool isDarkLane = (laneIndex >= 18.0);

    // Deterministic random numbers for this individual square tile
    float r1 = hash12(tileId);
    vec2 r2 = hash22(tileId + vec2(17.3, 41.7));

    // --- Tile Albedo & Grout Palette (matched to reference screenshots) ---
    vec3 tileColor;
    vec3 groutColor;
    vec3 bevelColor;

    if (!isDarkLane) {
      // Main Light-Blue Mosaic Zone:
      // - ~15% distinct darker cobalt/slate accent tiles (clearly visible in screenshots)
      // - ~10% slightly brighter sky-teal tiles
      // - ~75% base medium steel-teal tiles with subtle per-tile variation
      if (r1 < 0.155) {
        // Darker cobalt/slate blue square tile
        tileColor = mix(vec3(0.122, 0.272, 0.402), vec3(0.165, 0.330, 0.462), r2.x);
      } else if (r1 > 0.89) {
        // Slightly lighter teal-blue tile
        tileColor = mix(vec3(0.280, 0.462, 0.562), vec3(0.315, 0.500, 0.600), r2.x);
      } else {
        // Base medium pool blue tiles
        tileColor = mix(vec3(0.215, 0.390, 0.492), vec3(0.268, 0.448, 0.550), r2.x);
      }
      tileColor *= 0.97 + 0.05 * r2.y;

      groutColor = vec3(0.410, 0.560, 0.650);
      bevelColor = tileColor * 0.78;
    } else {
      // Dark Navy-Blue Lane Stripe Zone (right side in screenshots):
      if (r1 < 0.16) {
        // Deepest midnight navy tile
        tileColor = mix(vec3(0.025, 0.078, 0.142), vec3(0.040, 0.102, 0.175), r2.x);
      } else if (r1 > 0.86) {
        // Slightly richer Prussian navy tile
        tileColor = mix(vec3(0.078, 0.188, 0.298), vec3(0.102, 0.220, 0.340), r2.x);
      } else {
        // Base dark navy tile
        tileColor = mix(vec3(0.045, 0.120, 0.205), vec3(0.068, 0.160, 0.258), r2.x);
      }
      tileColor *= 0.96 + 0.06 * r2.y;

      // Crisp lighter steel-blue grout lines contrasting against the dark navy tiles
      groutColor = vec3(0.355, 0.485, 0.565);
      bevelColor = tileColor * 0.72;
    }

    // Compute distance from tile border (0.0 at edge of tile, 0.5 at center of tile)
    vec2 distToEdge2D = min(tileFract, 1.0 - tileFract);
    float distToEdge = min(distToEdge2D.x, distToEdge2D.y);

    // Grout line thickness (~2.3% half-width = 4.6% total grout line width)
    float groutHalfWidth = 0.023;
    float groutMask = 1.0 - smoothstep(groutHalfWidth - edgeAA, groutHalfWidth + edgeAA, distToEdge);

    // Subtle ceramic tile inner bevel & pillow shading
    float bevelMask = 1.0 - smoothstep(groutHalfWidth, groutHalfWidth + 0.055, distToEdge);
    float centerPillow = smoothstep(0.02, 0.38, distToEdge);

    vec3 ceramicSurface = mix(tileColor, bevelColor, bevelMask * 0.55);
    ceramicSurface *= (0.96 + 0.055 * centerPillow);

    // Combine ceramic tile and grout
    vec3 baseFloorColor = mix(ceramicSurface, groutColor, groutMask * 0.86);

    // --- Sample Physical Caustics (with 5-tap sub-pixel AA) & Underwater Bloom ---
    vec2 causticsUv = mix(vec2(0.10), vec2(0.90), vUv);
    vec2 dUv = uCausticsTexel * 0.65;

    vec4 c0 = texture2D(uCausticsTex, causticsUv);
    vec4 c1 = texture2D(uCausticsTex, causticsUv + vec2( dUv.x,  dUv.y));
    vec4 c2 = texture2D(uCausticsTex, causticsUv + vec2(-dUv.x,  dUv.y));
    vec4 c3 = texture2D(uCausticsTex, causticsUv + vec2( dUv.x, -dUv.y));
    vec4 c4 = texture2D(uCausticsTex, causticsUv + vec2(-dUv.x, -dUv.y));
    vec4 causticsRaw = c0 * 0.44 + (c1 + c2 + c3 + c4) * 0.14;

    vec4 bloomRaw = texture2D(uBloomTex, causticsUv);

    float sheetIrradiance = causticsRaw.r;
    float crispEdge = causticsRaw.g;
    float cuspHotspot = causticsRaw.b;
    float sheetCount = max(causticsRaw.a, 1.0);

    float bloomSheet = bloomRaw.r;
    float bloomEdge = bloomRaw.g;
    float bloomCusp = bloomRaw.b;

    // Map full pool depth uDepth in [0.0m, 2.0m] onto effective optical depth effDepth in [0.50m, 0.95m]
    float effDepth = mix(0.50, 0.95, clamp(uDepth / 2.0, 0.0, 1.0));

    // 1. Shallow vs. Deep Light Modulation:
    float shallowBoost = mix(1.55, 0.82, smoothstep(0.12, 0.55, effDepth));
    float dev = (sheetIrradiance - 1.0) * shallowBoost;

    // Background dimming in divergent troughs increases with depth as rays focus into lines
    float troughDimming = mix(0.95, 0.68, smoothstep(0.15, 0.85, effDepth));
    float baseIllum = troughDimming * clamp(1.0 + dev * 0.44, 0.50, 1.60);

    vec3 litTiles = baseFloorColor * baseIllum;

    // 2. Translucent Caustic Ribbon / Sheet Interior:
    float ribbonDepthFade = smoothstep(0.24, 0.78, effDepth);
    float ribbonInterior = max(0.0, sheetIrradiance - 1.06) * ribbonDepthFade;
    float foldedSheetLayers = max(0.0, sheetCount - 1.05) * smoothstep(0.70, 1.35, effDepth);

    // Soft logarithmic compression so multi-layered ribbons stay silky and translucent
    float translucentSheet = log(1.0 + ribbonInterior * 0.78) * 0.27 + foldedSheetLayers * 0.105;

    vec3 sheetTint = isDarkLane
      ? vec3(0.24, 0.65, 0.92)
      : vec3(0.38, 0.78, 0.96);

    // Multiply partly by baseFloorColor so darker blue accent tiles still show through the sheet!
    litTiles += (baseFloorColor * 0.75 + sheetTint * 0.55) * translucentSheet;

    // 3. Soft Underwater Scattered Cyan Glow (from bloom pass)
    float bloomIntensity = log(1.0 + max(0.0, bloomSheet - 1.05) * 0.35 + bloomEdge * 0.55 + bloomCusp * 0.75)
                           * smoothstep(0.18, 0.70, effDepth);
    vec3 glowColor = isDarkLane
      ? vec3(0.16, 0.52, 0.84)
      : vec3(0.26, 0.68, 0.90);
    litTiles += glowColor * bloomIntensity * 0.25;

    // 4. Razor-Sharp Caustic Fold Boundaries & Pinched Lines (crispEdge)
    float edgeStrength = 1.0 - exp(-crispEdge * 0.60);
    vec3 edgeColor = mix(
      vec3(0.54, 0.89, 1.0),
      vec3(0.96, 0.99, 1.0),
      clamp(crispEdge * 0.34, 0.0, 1.0)
    );
    litTiles += edgeColor * edgeStrength * (isDarkLane ? 0.68 : 0.62);

    // 5. Star Junctions & Cusp Vertices (cuspHotspot)
    float starStrength = 1.0 - exp(-cuspHotspot * 0.65);
    litTiles += vec3(0.95, 0.99, 1.0) * starStrength * 0.46;

    // Subtle tone mapping that preserves deep blues, tile contrast, and crisp white caustic edges
    vec3 finalColor = litTiles / (1.0 + max(vec3(0.0), litTiles - 0.82) * 0.35);
    finalColor = pow(clamp(finalColor, 0.0, 1.0), vec3(0.96));

    gl_FragColor = vec4(finalColor, 1.0);
  }
`;
