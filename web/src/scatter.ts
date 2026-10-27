// WebGL2 scatter plot for up to a few hundred thousand points, with
// drag-to-pan and wheel-to-zoom.

const VS = `#version 300 es
in vec2 a_pos;
in vec4 a_color;
uniform vec2 u_scale;
uniform vec2 u_offset;
uniform float u_size;
out vec4 v_color;
void main() {
  gl_Position = vec4(a_pos * u_scale + u_offset, 0.0, 1.0);
  gl_PointSize = u_size;
  v_color = a_color;
}`;

const FS = `#version 300 es
precision mediump float;
in vec4 v_color;
out vec4 outColor;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float r = dot(c, c);
  if (r > 1.0) discard;
  float edge = 1.0 - smoothstep(0.6, 1.0, r);
  outColor = vec4(v_color.rgb, v_color.a * edge);
}`;

function compile(gl: WebGL2RenderingContext, type: number, src: string) {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader");
  return s;
}

export class Scatter {
  private gl: WebGL2RenderingContext;
  private prog: WebGLProgram;
  private posBuf: WebGLBuffer;
  private colBuf: WebGLBuffer;
  private n = 0;
  private xy: Float32Array = new Float32Array();
  // view: data point (cx, cy) at the canvas centre, `zoom` canvas-heights per data unit
  private cx = 0;
  private cy = 0;
  private zoom = 1;
  private pending = false;
  // refit to every new set of positions until the user pans or zooms
  private follow = true;
  pointSize = 3;

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: true, premultipliedAlpha: false });
    if (!gl) throw new Error("WebGL2 is not available in this browser.");
    this.gl = gl;
    const p = gl.createProgram()!;
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(p);
    this.prog = p;
    this.posBuf = gl.createBuffer()!;
    this.colBuf = gl.createBuffer()!;
    new ResizeObserver(() => this.redraw()).observe(canvas);
    this.bindInput();
  }

  setPositions(xy: Float32Array, refit: boolean) {
    const gl = this.gl;
    this.xy = xy;
    this.n = xy.length / 2;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, xy, gl.DYNAMIC_DRAW);
    if (refit) this.follow = true;
    if (this.follow) this.fit();
    this.redraw();
  }

  /** RGBA bytes per point. */
  setColors(rgba: Uint8Array) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colBuf);
    gl.bufferData(gl.ARRAY_BUFFER, rgba, gl.DYNAMIC_DRAW);
    this.redraw();
  }

  clear() {
    this.n = 0;
    this.redraw();
  }

  fit() {
    if (this.n === 0) return;
    // robust bounds: 0.5% and 99.5% quantiles, so a few outliers do not shrink the view
    const xs = new Float32Array(this.n);
    const ys = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) {
      xs[i] = this.xy[2 * i];
      ys[i] = this.xy[2 * i + 1];
    }
    xs.sort();
    ys.sort();
    const q = (a: Float32Array, p: number) => a[Math.min(a.length - 1, Math.max(0, Math.round(p * (a.length - 1))))];
    const [x0, x1, y0, y1] = [q(xs, 0.005), q(xs, 0.995), q(ys, 0.005), q(ys, 0.995)];
    this.cx = (x0 + x1) / 2;
    this.cy = (y0 + y1) / 2;
    const w = this.canvas.clientWidth || 1;
    const h = this.canvas.clientHeight || 1;
    const span = Math.max((x1 - x0) * (h / w), y1 - y0, 1e-9);
    this.zoom = 0.9 / span;
  }

  redraw() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => {
      this.pending = false;
      this.draw();
    });
  }

  private draw() {
    const gl = this.gl;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (this.n === 0) return;
    gl.useProgram(this.prog);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    const sy = 2 * this.zoom;
    const sx = sy * (h / w);
    gl.uniform2f(gl.getUniformLocation(this.prog, "u_scale"), sx, sy);
    gl.uniform2f(gl.getUniformLocation(this.prog, "u_offset"), -this.cx * sx, -this.cy * sy);
    gl.uniform1f(gl.getUniformLocation(this.prog, "u_size"), this.pointSize * dpr);
    const aPos = gl.getAttribLocation(this.prog, "a_pos");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    const aCol = gl.getAttribLocation(this.prog, "a_color");
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colBuf);
    gl.enableVertexAttribArray(aCol);
    gl.vertexAttribPointer(aCol, 4, gl.UNSIGNED_BYTE, true, 0, 0);
    gl.drawArrays(gl.POINTS, 0, this.n);
  }

  private bindInput() {
    const c = this.canvas;
    let drag: { x: number; y: number } | null = null;
    c.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY };
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener("pointermove", (e) => {
      if (!drag) return;
      this.follow = false;
      const h = c.clientHeight || 1;
      this.cx -= (e.clientX - drag.x) / (h * this.zoom);
      this.cy += (e.clientY - drag.y) / (h * this.zoom);
      drag = { x: e.clientX, y: e.clientY };
      this.redraw();
    });
    c.addEventListener("pointerup", () => (drag = null));
    c.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.follow = false;
        const r = c.getBoundingClientRect();
        const h = r.height || 1;
        // keep the data point under the cursor fixed
        const mx = (e.clientX - r.left - r.width / 2) / h;
        const my = -(e.clientY - r.top - r.height / 2) / h;
        const before = [this.cx + mx / this.zoom, this.cy + my / this.zoom];
        this.zoom *= Math.exp(-e.deltaY * 0.0015);
        this.cx = before[0] - mx / this.zoom;
        this.cy = before[1] - my / this.zoom;
        this.redraw();
      },
      { passive: false },
    );
    c.addEventListener("dblclick", () => {
      this.follow = true;
      this.fit();
      this.redraw();
    });
  }
}
