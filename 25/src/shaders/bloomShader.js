// Separable Gaussian / Underwater Light Scattering Bloom Shader
// Creates the soft cyan-white glow around intense caustic lines and star junctions.

export const bloomVertexShader = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

export const bloomFragmentShader = /* glsl */ `
  precision highp float;

  uniform sampler2D uTexture;
  uniform vec2 uDirection; // (1.0 / width, 0.0) or (0.0, 1.0 / height)

  varying vec2 vUv;

  void main() {
    // 9-tap Gaussian filter using hardware bilinear filtering
    vec4 sum = vec4(0.0);

    vec2 off1 = vec2(1.411764705882353) * uDirection;
    vec2 off2 = vec2(3.2941176470588234) * uDirection;
    vec2 off3 = vec2(5.176470588235294) * uDirection;
    vec2 off4 = vec2(7.0588235294117645) * uDirection;

    sum += texture2D(uTexture, vUv) * 0.1964825501511404;
    sum += texture2D(uTexture, vUv + off1) * 0.24690696483280675;
    sum += texture2D(uTexture, vUv - off1) * 0.24690696483280675;
    sum += texture2D(uTexture, vUv + off2) * 0.10699735302869422;
    sum += texture2D(uTexture, vUv - off2) * 0.10699735302869422;
    sum += texture2D(uTexture, vUv + off3) * 0.03892263569953543;
    sum += texture2D(uTexture, vUv - off3) * 0.03892263569953543;
    sum += texture2D(uTexture, vUv + off4) * 0.008931833267746368;
    sum += texture2D(uTexture, vUv - off4) * 0.008931833267746368;

    gl_FragColor = sum;
  }
`;
