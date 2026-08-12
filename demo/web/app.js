"use strict";

const $ = (selector) => document.querySelector(selector);
const state = { models: [], selected: null, shownAnimation: null, poll: null, uploadBusy: false };

function toast(message, error = false) {
  const box = $("#toast");
  box.textContent = message;
  box.className = "toast show" + (error ? " error" : "");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { box.className = "toast"; }, 3600);
}

async function api(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  if (!response.ok) {
    const message = (await response.text()).trim();
    throw new Error(message || `Request failed (${response.status})`);
  }
  if (response.status === 204) return null;
  return response.json();
}

function formatBytes(value) {
  if (value < 1024) return `${value} B`;
  const units = ["KiB", "MiB", "GiB"];
  let amount = value / 1024;
  let unit = 0;
  while (amount >= 1024 && unit < units.length - 1) { amount /= 1024; unit++; }
  return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${units[unit]}`;
}

function formatDate(value) {
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "0:00";
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

class GLBViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2", { antialias: true, alpha: true, preserveDrawingBuffer: true });
    if (!this.gl) throw new Error("This browser does not support WebGL 2");
    this.base = null;
    this.morphs = [];
    this.times = null;
    this.weights = null;
    this.duration = 0;
    this.current = 0;
    this.playing = false;
    this.yaw = .65;
    this.pitch = .22;
    this.distance = 3;
    this.target = [0, 0, 0];
    this.radius = 1;
    this.loadToken = 0;
    this.dirtyGeometry = false;
    this.lastFrame = performance.now();
    this.initGL();
    this.bindControls();
    new ResizeObserver(() => this.resize()).observe(canvas);
    requestAnimationFrame((time) => this.frame(time));
  }

  initGL() {
    const gl = this.gl;
    const vertex = `#version 300 es
      precision highp float;
      layout(location=0) in vec3 aPosition;
      layout(location=1) in vec3 aNormal;
      uniform mat4 uMVP;
      out vec3 vNormal;
      out vec3 vPosition;
      void main(){ vNormal=aNormal; vPosition=aPosition; gl_Position=uMVP*vec4(aPosition,1.0); }`;
    const fragment = `#version 300 es
      precision highp float;
      in vec3 vNormal;
      in vec3 vPosition;
      out vec4 color;
      void main(){
        vec3 n=normalize(vNormal);
        vec3 light=normalize(vec3(.45,.8,.65));
        float diffuse=max(dot(n,light),0.0);
        float rim=pow(1.0-max(abs(n.z),0.0),2.2);
        vec3 low=vec3(.055,.26,.31), high=vec3(.25,.86,.77);
        vec3 c=mix(low,high,.24+diffuse*.7)+rim*vec3(.23,.12,.38);
        color=vec4(c,1.0);
      }`;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    this.program = gl.createProgram();
    gl.attachShader(this.program, compile(gl.VERTEX_SHADER, vertex));
    gl.attachShader(this.program, compile(gl.FRAGMENT_SHADER, fragment));
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(this.program));
    this.mvpLocation = gl.getUniformLocation(this.program, "uMVP");
    this.vao = gl.createVertexArray();
    this.positionBuffer = gl.createBuffer();
    this.normalBuffer = gl.createBuffer();
    this.indexBuffer = gl.createBuffer();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.normalBuffer);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bindVertexArray(null);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
  }

  bindControls() {
    let dragging = false, x = 0, y = 0;
    this.canvas.addEventListener("pointerdown", (event) => {
      dragging = true; x = event.clientX; y = event.clientY;
      this.canvas.setPointerCapture(event.pointerId);
    });
    this.canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      this.yaw += (event.clientX - x) * .008;
      this.pitch = Math.max(-1.35, Math.min(1.35, this.pitch + (event.clientY - y) * .008));
      x = event.clientX; y = event.clientY;
    });
    this.canvas.addEventListener("pointerup", () => { dragging = false; });
    this.canvas.addEventListener("pointercancel", () => { dragging = false; });
    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.distance = Math.max(this.radius * 1.05, Math.min(this.radius * 12, this.distance * Math.exp(event.deltaY * .001)));
    }, { passive: false });
  }

  resetCamera() {
    this.yaw = .65; this.pitch = .22; this.distance = this.radius * 3;
  }

  resize() {
    const ratio = Math.min(devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(this.canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(this.canvas.clientHeight * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width; this.canvas.height = height;
    }
  }

  clear() {
    this.loadToken++;
    this.base = null; this.morphs = []; this.times = null; this.weights = null;
    this.duration = 0; this.current = 0; this.playing = false;
    updateTimeline();
  }

  async load(url) {
    const token = ++this.loadToken;
    this.playing = false;
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(`Could not load GLB (${response.status})`);
    const buffer = await response.arrayBuffer();
    if (token !== this.loadToken) return false;
    const mesh = parseGLB(buffer);
    if (token !== this.loadToken) return false;
    this.base = mesh.positions;
    this.indices = mesh.indices;
    this.morphs = mesh.morphs;
    this.times = mesh.times;
    this.weights = mesh.weights;
    this.duration = mesh.duration;
    this.current = 0;
    this.fit(mesh.positions);
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.indices, gl.STATIC_DRAW);
    this.dirtyGeometry = true;
    this.uploadGeometry();
    updateTimeline();
    return true;
  }

  fit(positions) {
    const minimum = [Infinity, Infinity, Infinity], maximum = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < positions.length; i += 3) {
      for (let axis = 0; axis < 3; axis++) {
        minimum[axis] = Math.min(minimum[axis], positions[i + axis]);
        maximum[axis] = Math.max(maximum[axis], positions[i + axis]);
      }
    }
    this.target = minimum.map((v, i) => (v + maximum[i]) * .5);
    this.radius = Math.max(.001, Math.hypot(maximum[0]-minimum[0], maximum[1]-minimum[1], maximum[2]-minimum[2]) * .5);
    this.resetCamera();
  }

  sampledPositions() {
    if (!this.morphs.length || !this.times || !this.times.length) return this.base;
    let upper = 1;
    while (upper < this.times.length && this.times[upper] < this.current) upper++;
    upper = Math.min(upper, this.times.length - 1);
    const lower = Math.max(0, upper - 1);
    const span = this.times[upper] - this.times[lower];
    const alpha = span > 0 ? (this.current - this.times[lower]) / span : 0;
    const result = new Float32Array(this.base);
    const targetCount = this.morphs.length;
    for (let target = 0; target < targetCount; target++) {
      const a = this.weights[lower * targetCount + target] || 0;
      const b = this.weights[upper * targetCount + target] || 0;
      const weight = a + (b - a) * alpha;
      if (Math.abs(weight) < 1e-6) continue;
      const delta = this.morphs[target];
      for (let i = 0; i < result.length; i++) result[i] += delta[i] * weight;
    }
    return result;
  }

  uploadGeometry() {
    if (!this.base || !this.dirtyGeometry) return;
    const positions = this.sampledPositions();
    const normals = computeNormals(positions, this.indices);
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, normals, gl.DYNAMIC_DRAW);
    this.dirtyGeometry = false;
  }

  seek(value) {
    this.current = Math.max(0, Math.min(this.duration, value));
    this.dirtyGeometry = true;
    updateTimeline();
  }

  frame(now) {
    const elapsed = Math.min(.1, Math.max(0, (now - this.lastFrame) / 1000));
    this.lastFrame = now;
    if (this.playing && this.duration > 0) {
      this.current += elapsed;
      if (this.current >= this.duration) this.current %= this.duration;
      this.dirtyGeometry = true;
      updateTimeline();
    }
    this.resize(); this.uploadGeometry(); this.draw();
    requestAnimationFrame((time) => this.frame(time));
  }

  draw() {
    const gl = this.gl;
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.base) return;
    const cp = Math.cos(this.pitch), eye = [
      this.target[0] + this.distance * cp * Math.sin(this.yaw),
      this.target[1] + this.distance * Math.sin(this.pitch),
      this.target[2] + this.distance * cp * Math.cos(this.yaw),
    ];
    const projection = perspective(.78, this.canvas.width / this.canvas.height, Math.max(.0001, this.radius / 100), this.radius * 30);
    const view = lookAt(eye, this.target, [0, 1, 0]);
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.mvpLocation, false, multiply4(projection, view));
    gl.bindVertexArray(this.vao);
    gl.drawElements(gl.TRIANGLES, this.indices.length, gl.UNSIGNED_INT, 0);
  }
}

function parseGLB(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength < 20 || view.getUint32(0, true) !== 0x46546c67 || view.getUint32(4, true) !== 2 || view.getUint32(8, true) !== buffer.byteLength) {
    throw new Error("Invalid GLB 2.0 file");
  }
  let offset = 12, json = null, binary = null;
  while (offset + 8 <= buffer.byteLength) {
    const length = view.getUint32(offset, true), type = view.getUint32(offset + 4, true);
    offset += 8;
    if (offset + length > buffer.byteLength) throw new Error("Truncated GLB chunk");
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, offset, length)).replace(/\0+$/g, ""));
    if (type === 0x004e4942) binary = buffer.slice(offset, offset + length);
    offset += length;
  }
  if (!json || !binary) throw new Error("The GLB needs JSON and binary chunks");
  const components = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 };
  const sizes = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
  const readComponent = (data, byte, type) => {
    switch (type) {
      case 5120: return data.getInt8(byte); case 5121: return data.getUint8(byte);
      case 5122: return data.getInt16(byte, true); case 5123: return data.getUint16(byte, true);
      case 5125: return data.getUint32(byte, true); case 5126: return data.getFloat32(byte, true);
      default: throw new Error(`Unsupported accessor component type ${type}`);
    }
  };
  const normalize = (value, type) => type === 5120 ? Math.max(value / 127, -1) : type === 5121 ? value / 255 : type === 5122 ? Math.max(value / 32767, -1) : type === 5123 ? value / 65535 : value;
  const accessor = (index) => {
    const spec = json.accessors?.[index];
    if (!spec || spec.bufferView === undefined || spec.sparse) throw new Error("Sparse or missing accessors are not supported by this preview");
    const bv = json.bufferViews?.[spec.bufferView];
    if (!bv || (bv.buffer ?? 0) !== 0) throw new Error("Invalid GLB buffer view");
    const count = spec.count, width = components[spec.type], size = sizes[spec.componentType];
    if (!Number.isSafeInteger(count) || !width || !size || count <= 0) throw new Error("Invalid GLB accessor");
    const stride = bv.byteStride || width * size;
    const viewStart = bv.byteOffset || 0, viewEnd = viewStart + bv.byteLength;
    const start = viewStart + (spec.byteOffset || 0);
    const end = start + Math.max(0, count - 1) * stride + width * size;
    if (!Number.isSafeInteger(viewStart) || !Number.isSafeInteger(bv.byteLength) || !Number.isSafeInteger(stride) || !Number.isSafeInteger(start) || !Number.isSafeInteger(end) || stride < width * size || start < viewStart || end > viewEnd || viewEnd > binary.byteLength) throw new Error("GLB accessor is out of bounds");
    const data = new DataView(binary);
    const result = new Float64Array(count * width);
    for (let row = 0; row < count; row++) for (let column = 0; column < width; column++) {
      let value = readComponent(data, start + row * stride + column * size, spec.componentType);
      if (spec.normalized) value = normalize(value, spec.componentType);
      result[row * width + column] = value;
    }
    return result;
  };

  const parts = [];
  const appendMesh = (meshIndex, transform) => {
    const mesh = json.meshes?.[meshIndex];
    if (!mesh) return;
    for (const primitive of mesh.primitives || []) {
      if ((primitive.mode ?? 4) !== 4 || primitive.attributes?.POSITION === undefined) continue;
      const raw = accessor(primitive.attributes.POSITION);
      if (raw.length % 3 || raw.some((value) => !Number.isFinite(value))) throw new Error("Mesh positions are invalid");
      const positions = new Float32Array(raw.length);
      for (let i = 0; i < raw.length; i += 3) transformPoint(transform, raw, i, positions);
      let indices;
      if (primitive.indices !== undefined) {
        const source = accessor(primitive.indices); indices = new Uint32Array(source.length);
        if (source.length % 3) throw new Error("Triangle index count is invalid");
        for (let i = 0; i < source.length; i++) {
          if (!Number.isSafeInteger(source[i]) || source[i] < 0 || source[i] >= positions.length / 3) throw new Error("Mesh index is out of bounds");
          indices[i] = source[i];
        }
      } else {
        indices = new Uint32Array(positions.length / 3);
        for (let i = 0; i < indices.length; i++) indices[i] = i;
      }
      const morphs = (primitive.targets || []).map((target) => {
        if (target.POSITION === undefined) return new Float32Array(positions.length);
        const rawDelta = accessor(target.POSITION), delta = new Float32Array(rawDelta.length);
        if (rawDelta.length !== positions.length || rawDelta.some((value) => !Number.isFinite(value))) throw new Error("Morph target positions are invalid");
        for (let i = 0; i < rawDelta.length; i += 3) transformDelta(transform, rawDelta, i, delta);
        return delta;
      });
      parts.push({ positions, indices, morphs });
    }
  };

  const roots = json.scenes?.[json.scene || 0]?.nodes;
  if (roots?.length && json.nodes) {
    const visit = (nodeIndex, parent, depth) => {
      if (depth > 128) throw new Error("GLB scene graph is too deep");
      const node = json.nodes[nodeIndex]; if (!node) return;
      const world = multiply4(parent, nodeMatrix(node));
      if (node.mesh !== undefined) appendMesh(node.mesh, world);
      for (const child of node.children || []) visit(child, world, depth + 1);
    };
    for (const root of roots) visit(root, identity4(), 0);
  } else {
    for (let index = 0; index < (json.meshes || []).length; index++) appendMesh(index, identity4());
  }
  if (!parts.length) throw new Error("No triangle meshes found in this GLB");
  const vertexCount = parts.reduce((sum, part) => sum + part.positions.length / 3, 0);
  const indexCount = parts.reduce((sum, part) => sum + part.indices.length, 0);
  if (vertexCount > 2_000_000 || indexCount > 12_000_000) throw new Error("Mesh is too large for the browser preview");
  const positions = new Float32Array(vertexCount * 3), indices = new Uint32Array(indexCount);
  const morphCount = Math.max(0, ...parts.map((part) => part.morphs.length));
  const morphs = Array.from({ length: morphCount }, () => new Float32Array(vertexCount * 3));
  let vertexOffset = 0, indexOffset = 0;
  for (const part of parts) {
    positions.set(part.positions, vertexOffset * 3);
    for (let i = 0; i < part.indices.length; i++) indices[indexOffset + i] = part.indices[i] + vertexOffset;
    for (let target = 0; target < part.morphs.length; target++) morphs[target].set(part.morphs[target], vertexOffset * 3);
    vertexOffset += part.positions.length / 3; indexOffset += part.indices.length;
  }

  let times = null, weights = null, duration = 0;
  outer: for (const animation of json.animations || []) for (const channel of animation.channels || []) {
    if (channel.target?.path !== "weights") continue;
    const sampler = animation.samplers?.[channel.sampler];
    if (!sampler || sampler.interpolation && sampler.interpolation !== "LINEAR" && sampler.interpolation !== "STEP") continue;
    const input = accessor(sampler.input), output = accessor(sampler.output);
    if (!morphCount || output.length < input.length * morphCount) continue;
    if (input.some((value, index) => !Number.isFinite(value) || value < 0 || index > 0 && value < input[index - 1]) || output.some((value) => !Number.isFinite(value))) throw new Error("Animation samples are invalid");
    times = Float32Array.from(input); weights = Float32Array.from(output.slice(0, input.length * morphCount));
    duration = times.length ? times[times.length - 1] : 0;
    break outer;
  }
  return { positions, indices, morphs, times, weights, duration };
}

function identity4() { return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]); }
function multiply4(a, b) {
  const out = new Float32Array(16);
  for (let column=0; column<4; column++) for (let row=0; row<4; row++) {
    let value=0; for (let k=0; k<4; k++) value += a[k*4+row]*b[column*4+k];
    out[column*4+row]=value;
  }
  return out;
}
function nodeMatrix(node) {
  if (node.matrix?.length === 16) {
    const matrix=Float32Array.from(node.matrix);
    if(matrix.some((value)=>!Number.isFinite(value)))throw new Error("Node matrix is invalid");
    return matrix;
  }
  const [x,y,z,w] = node.rotation || [0,0,0,1], [sx,sy,sz] = node.scale || [1,1,1], [tx,ty,tz] = node.translation || [0,0,0];
  const x2=x+x,y2=y+y,z2=z+z, xx=x*x2,xy=x*y2,xz=x*z2, yy=y*y2,yz=y*z2,zz=z*z2, wx=w*x2,wy=w*y2,wz=w*z2;
  const matrix=new Float32Array([
    (1-(yy+zz))*sx,(xy+wz)*sx,(xz-wy)*sx,0,
    (xy-wz)*sy,(1-(xx+zz))*sy,(yz+wx)*sy,0,
    (xz+wy)*sz,(yz-wx)*sz,(1-(xx+yy))*sz,0,
    tx,ty,tz,1,
  ]);
  if(matrix.some((value)=>!Number.isFinite(value)))throw new Error("Node transform is invalid");
  return matrix;
}
function transformPoint(matrix, source, index, output) {
  const x=source[index],y=source[index+1],z=source[index+2];
  output[index]=matrix[0]*x+matrix[4]*y+matrix[8]*z+matrix[12];
  output[index+1]=matrix[1]*x+matrix[5]*y+matrix[9]*z+matrix[13];
  output[index+2]=matrix[2]*x+matrix[6]*y+matrix[10]*z+matrix[14];
}
function transformDelta(matrix, source, index, output) {
  const x=source[index],y=source[index+1],z=source[index+2];
  output[index]=matrix[0]*x+matrix[4]*y+matrix[8]*z;
  output[index+1]=matrix[1]*x+matrix[5]*y+matrix[9]*z;
  output[index+2]=matrix[2]*x+matrix[6]*y+matrix[10]*z;
}
function perspective(fov, aspect, near, far) {
  const f=1/Math.tan(fov/2), nf=1/(near-far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}
function lookAt(eye, center, up) {
  let zx=eye[0]-center[0],zy=eye[1]-center[1],zz=eye[2]-center[2]; let n=1/Math.hypot(zx,zy,zz);zx*=n;zy*=n;zz*=n;
  let xx=up[1]*zz-up[2]*zy,xy=up[2]*zx-up[0]*zz,xz=up[0]*zy-up[1]*zx;n=1/Math.max(1e-12,Math.hypot(xx,xy,xz));xx*=n;xy*=n;xz*=n;
  const yx=zy*xz-zz*xy,yy=zz*xx-zx*xz,yz=zx*xy-zy*xx;
  return new Float32Array([xx,yx,zx,0, xy,yy,zy,0, xz,yz,zz,0, -(xx*eye[0]+xy*eye[1]+xz*eye[2]),-(yx*eye[0]+yy*eye[1]+yz*eye[2]),-(zx*eye[0]+zy*eye[1]+zz*eye[2]),1]);
}
function computeNormals(positions, indices) {
  const normals=new Float32Array(positions.length);
  for(let i=0;i+2<indices.length;i+=3){const a=indices[i]*3,b=indices[i+1]*3,c=indices[i+2]*3;if(c+2>=positions.length)continue;const abx=positions[b]-positions[a],aby=positions[b+1]-positions[a+1],abz=positions[b+2]-positions[a+2],acx=positions[c]-positions[a],acy=positions[c+1]-positions[a+1],acz=positions[c+2]-positions[a+2],nx=aby*acz-abz*acy,ny=abz*acx-abx*acz,nz=abx*acy-aby*acx;for(const p of[a,b,c]){normals[p]+=nx;normals[p+1]+=ny;normals[p+2]+=nz;}}
  for(let i=0;i<normals.length;i+=3){const n=1/Math.max(1e-12,Math.hypot(normals[i],normals[i+1],normals[i+2]));normals[i]*=n;normals[i+1]*=n;normals[i+2]*=n;} return normals;
}

let viewer;
function updateTimeline() {
  if (!viewer) return;
  const enabled = viewer.duration > 0;
  $("#timeline").classList.toggle("disabled", !enabled);
  $("#scrubber").max = String(Math.max(viewer.duration, .001));
  $("#scrubber").value = String(viewer.current);
  $("#currentTime").textContent = formatDuration(viewer.current);
  $("#duration").textContent = formatDuration(viewer.duration);
  $("#play").textContent = viewer.playing ? "❚❚" : "▶";
}

async function loadViewer(url, title, kind, animationID = null, saveThumbnail = false) {
  $("#viewerLoading").classList.remove("hidden"); $("#viewerError").classList.add("hidden");
  $("#stageEmpty").classList.add("hidden"); $("#viewport").classList.add("has-model");
  $("#viewerTitle").textContent = title; $("#viewerKind").textContent = kind;
  try {
    const loaded = await viewer.load(url);
    if (!loaded) return;
    state.shownAnimation = animationID;
    $("#resetView").disabled = false; $("#showSource").disabled = !state.selected;
    renderAnimations();
    if (saveThumbnail && state.selected && !state.selected.thumbnail) {
      setTimeout(async () => {
        try {
          const thumbnail = viewer.canvas.toDataURL("image/webp", .72);
          await api(`/api/model/${state.selected.id}/thumbnail`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({thumbnail}) });
          await refreshModels(false);
        } catch (_) { /* A thumbnail is optional. */ }
      }, 180);
    }
  } catch (error) {
    $("#viewerError").textContent = error.message;
    $("#viewerError").classList.remove("hidden");
  } finally { $("#viewerLoading").classList.add("hidden"); }
}

function renderModels() {
  const grid = $("#modelGrid"); grid.replaceChildren();
  $("#modelCount").textContent = state.models.length;
  $("#libraryEmpty").classList.toggle("hidden", state.models.length !== 0);
  for (const model of state.models) {
    const card = el("button", "model-card" + (state.selected?.id === model.id ? " active" : ""));
    card.type = "button";
    const thumb = el("span", "model-thumb");
    if (model.thumbnail) { const image=el("img"); image.src=model.thumbnail; image.alt=""; thumb.append(image); }
    else thumb.append(el("span", "", "◇"));
    const copy=el("span","model-card-copy"), name=el("strong","",model.name), meta=el("small","",`${model.animationCount} animation${model.animationCount===1?"":"s"}`);
    copy.append(name,meta); card.append(thumb,copy); card.addEventListener("click",()=>selectModel(model.id)); grid.append(card);
  }
}

function renderSelection() {
  const selected = state.selected;
  $("#noSelection").classList.toggle("hidden", !!selected);
  $("#selection").classList.toggle("hidden", !selected);
  if (!selected) return;
  $("#modelName").textContent = selected.name;
  $("#modelMeta").textContent = `${formatBytes(selected.sizeBytes)} · added ${formatDate(selected.createdAt)}`;
  $("#animationCount").textContent = selected.animations.length;
  $("#historyEmpty").classList.toggle("hidden", selected.animations.length !== 0);
  renderAnimations();
}

function renderAnimations() {
  const list=$("#animationList"); list.replaceChildren();
  if (!state.selected) return;
  for(const animation of state.selected.animations){
    const card=el("article","animation-card"+(state.shownAnimation===animation.id?" active":""));
    const top=el("div","animation-card-top"), prompt=el("p","",animation.options.prompt);
    top.append(prompt); card.append(top);
    const backend=animation.backend||"backend";
    card.append(el("small","",`${formatDate(animation.createdAt)} · ${(animation.durationMs/1000).toFixed(2)}s · ${backend}`));
    const actions=el("div","animation-actions"), view=el("button","view","▶ View"), download=el("a","","↓ GLB"), remove=el("button","remove","×");
    view.type="button"; view.addEventListener("click",()=>loadViewer(`/api/animation/${animation.id}`,animation.options.prompt,"Saved animation",animation.id));
    download.href=`/api/animation/${animation.id}?download=1`; download.download=`animation-${animation.id}.glb`; download.title="Download GLB";
    remove.type="button"; remove.title="Delete animation"; remove.addEventListener("click",()=>deleteAnimation(animation));
    actions.append(view,download,remove);card.append(actions);list.append(card);
  }
}

async function refreshModels(render = true) {
  state.models = await api("/api/models");
  if (state.selected) {
    const summary = state.models.find((item) => item.id === state.selected.id);
    if (summary) state.selected.thumbnail = summary.thumbnail;
  }
  if (render) renderModels(); else renderModels();
}

async function selectModel(id) {
  try {
    state.selected = await api(`/api/model/${id}`); state.shownAnimation=null;
    renderModels(); renderSelection();
    await loadViewer(`/api/model/${id}/source`, state.selected.name, "Original model", null, true);
  } catch(error){ toast(error.message,true); }
}

function uploadModel(file) {
  if(state.uploadBusy || !file)return;
  if(!file.name.toLowerCase().endsWith(".glb")){toast("Please choose a binary .glb model",true);return;}
  state.uploadBusy=true; const data=new FormData();data.append("model",file,file.name);const request=new XMLHttpRequest();
  request.open("POST","/api/models");
  request.upload.onprogress=(event)=>{if(event.lengthComputable)$("#uploadProgress").style.width=`${event.loaded/event.total*100}%`;};
  request.onload=async()=>{state.uploadBusy=false;$("#uploadProgress").style.width="0";if(request.status!==201){toast(request.responseText.trim()||"Upload failed",true);return;}try{const model=JSON.parse(request.responseText);await refreshModels();await selectModel(model.id);toast(`${model.name} joined the gallery`);}catch(error){toast(error.message,true);}};
  request.onerror=()=>{state.uploadBusy=false;$("#uploadProgress").style.width="0";toast("Upload failed",true);}; request.send(data);
}

async function generateAnimation(event) {
  event.preventDefault(); if(!state.selected)return;
  const button=$("#generate");button.disabled=true;
  try{
    const body=new URLSearchParams(new FormData(event.currentTarget));
    const queued=await api(`/api/model/${state.selected.id}/animations`,{method:"POST",headers:{"Content-Type":"application/x-www-form-urlencoded"},body});
    showJob("queued","Waiting for the inference worker…");pollJob(queued.job, state.selected.id);
  }catch(error){button.disabled=false;toast(error.message,true);}
}

function showJob(status, detail, error=false) {
  const job=$("#job");job.classList.remove("hidden");job.classList.toggle("error",error);$("#jobTitle").textContent=status;$("#jobDetail").textContent=detail;
}

function pollJob(id, modelID) {
  clearInterval(state.poll);
  const check=async()=>{
    try{
      const job=await api(`/api/job/${id}`);
      if(job.state==="queued")showJob("Queued","Waiting for the inference worker…");
      if(job.state==="running")showJob("Animating locally",`Running ${job.options.steps} diffusion steps on the configured backend…`);
      if(job.state==="error"){clearInterval(state.poll);state.poll=null;$("#generate").disabled=false;showJob("Generation failed",job.error,true);return;}
      if(job.state==="done"){
        clearInterval(state.poll);state.poll=null;$("#generate").disabled=false;showJob("Animation saved",`${(job.durationMs/1000).toFixed(2)}s · ${job.backend||"configured backend"}`);
        await refreshModels();
        if(state.selected?.id===modelID){state.selected=await api(`/api/model/${modelID}`);renderSelection();await loadViewer(`/api/animation/${id}`,job.options.prompt,"Saved animation",id);}
        else toast("Animation saved to its model gallery");
        setTimeout(()=>$("#job").classList.add("hidden"),2200);
      }
    }catch(error){clearInterval(state.poll);state.poll=null;$("#generate").disabled=false;showJob("Could not track job",error.message,true);}
  }; check(); state.poll=setInterval(check,850);
}

async function deleteAnimation(animation) {
  if(!confirm("Delete this saved animation?"))return;
  try{await api(`/api/animation/${animation.id}`,{method:"DELETE"});state.selected=await api(`/api/model/${state.selected.id}`);if(state.shownAnimation===animation.id){state.shownAnimation=null;await loadViewer(`/api/model/${state.selected.id}/source`,state.selected.name,"Original model");}await refreshModels();renderSelection();toast("Animation deleted");}catch(error){toast(error.message,true);}
}

async function deleteSelectedModel() {
  if(!state.selected||!confirm(`Delete ${state.selected.name} and all of its animations?`))return;
  const id=state.selected.id;
  try{await api(`/api/model/${id}`,{method:"DELETE"});state.selected=null;state.shownAnimation=null;viewer.clear();$("#stageEmpty").classList.remove("hidden");$("#viewport").classList.remove("has-model");$("#viewerTitle").textContent="No model selected";$("#viewerKind").textContent="Preview";$("#resetView").disabled=true;$("#showSource").disabled=true;renderSelection();await refreshModels();toast("Model and its animations deleted");}catch(error){toast(error.message,true);}
}

async function init() {
  try { viewer=new GLBViewer($("#viewer")); }
  catch(error){$("#viewerError").textContent=error.message;$("#viewerError").classList.remove("hidden");return;}
  $("#fileInput").addEventListener("change",(event)=>uploadModel(event.target.files[0]));
  const drop=$("#dropzone");
  for(const type of["dragenter","dragover"]){drop.addEventListener(type,(event)=>{event.preventDefault();drop.classList.add("drag");});}
  for(const type of["dragleave","drop"]){drop.addEventListener(type,(event)=>{event.preventDefault();drop.classList.remove("drag");});}
  drop.addEventListener("drop",(event)=>uploadModel(event.dataTransfer.files[0]));
  $("#animationForm").addEventListener("submit",generateAnimation);
  $("#deleteModel").addEventListener("click",deleteSelectedModel);
  $("#resetView").addEventListener("click",()=>viewer.resetCamera());
  $("#showSource").addEventListener("click",()=>state.selected&&loadViewer(`/api/model/${state.selected.id}/source`,state.selected.name,"Original model"));
  $("#play").addEventListener("click",()=>{viewer.playing=!viewer.playing;updateTimeline();});
  $("#scrubber").addEventListener("input",(event)=>viewer.seek(Number(event.target.value)));
  try{
    const [info,models]=await Promise.all([api("/api/info"),api("/api/models")]);state.models=models;
    $("#runtime").classList.add("ready");$("#runtime span").textContent=`${info.device} · ${info.model}`;$("#uploadLimit").textContent=`${info.maxUploadMiB} MiB max`;renderModels();
    if(models.length)await selectModel(models[0].id);
  }catch(error){$("#runtime span").textContent="Server unavailable";toast(error.message,true);}
}

init();
