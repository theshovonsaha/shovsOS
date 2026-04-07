import { useEffect, useRef } from 'react'
import * as THREE from 'three'

type OrbPhase = 'idle' | 'thinking' | 'working' | 'finalizing'

interface AgentOrbProps {
  phase: OrbPhase
  size?: number
}

// ─── Orb vertex shader — displaces sphere vertices with time-driven noise ────
const vertexShader = `
uniform float uTime;
uniform float uTurbulence;

varying vec3 vNormal;
varying vec3 vPosition;
varying float vDisplacement;

float hash(vec3 p) {
  p = fract(p * vec3(443.897, 441.423, 437.195));
  p += dot(p, p.yxz + 19.19);
  return fract((p.x + p.y) * p.z);
}

float noise(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(mix(hash(i), hash(i+vec3(1,0,0)), f.x),
        mix(hash(i+vec3(0,1,0)), hash(i+vec3(1,1,0)), f.x), f.y),
    mix(mix(hash(i+vec3(0,0,1)), hash(i+vec3(1,0,1)), f.x),
        mix(hash(i+vec3(0,1,1)), hash(i+vec3(1,1,1)), f.x), f.y),
    f.z
  );
}

float fbm(vec3 p) {
  float v = 0.0;
  float a = 0.5;
  for(int i = 0; i < 4; i++) {
    v += a * noise(p);
    p = p * 2.1 + vec3(1.7, 9.2, 8.3);
    a *= 0.5;
  }
  return v;
}

void main() {
  vNormal = normal;
  vPosition = position;

  float n = fbm(position * 2.5 + uTime * 0.22);
  float displacement = n * uTurbulence * 0.28;
  vDisplacement = displacement;

  vec3 displaced = position + normal * displacement;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
}
`

// ─── Orb fragment shader — phase-aware color field with fresnel rim ──────────
const fragmentShader = `
uniform float uTime;
uniform float uTurbulence;
uniform vec3 uColorCore;
uniform vec3 uColorRim;
uniform vec3 uColorMid;

varying vec3 vNormal;
varying vec3 vPosition;
varying float vDisplacement;

void main() {
  vec3 viewDir = normalize(cameraPosition - vPosition);
  float fresnel = pow(1.0 - max(dot(normalize(vNormal), viewDir), 0.0), 3.2);

  float heat = vDisplacement * 3.5;
  vec3 col = mix(uColorCore, uColorMid, heat);
  col = mix(col, uColorRim, fresnel * 0.85);

  float shimmer = sin(vPosition.y * 12.0 + uTime * 1.8) * 0.04;
  col += shimmer;

  float alpha = 0.92 + fresnel * 0.08;
  gl_FragColor = vec4(col, alpha);
}
`

// ─── Particle vertex shader — each point orbits with unique speed/radius ─────
const particleVertexShader = `
uniform float uTime;
uniform float uSpeed;
uniform float uSpread;

attribute float aRadius;
attribute float aTheta;
attribute float aPhi;
attribute float aSpeed;
attribute float aSize;

varying float vAlpha;

void main() {
  float t = uTime * aSpeed * uSpeed;

  // Orbit — each particle has its own radius, latitude, longitude
  float theta = aTheta + t * 0.6;
  float phi   = aPhi   + sin(t * 0.3 + aTheta) * 0.4;
  float r     = aRadius * uSpread;

  vec3 pos = vec3(
    r * sin(phi) * cos(theta),
    r * cos(phi),
    r * sin(phi) * sin(theta)
  );

  // Gentle drift
  pos.x += sin(t * 1.2 + aTheta * 3.0) * 0.08;
  pos.y += cos(t * 0.9 + aPhi   * 2.0) * 0.08;

  vec4 mvPos = modelViewMatrix * vec4(pos, 1.0);
  gl_PointSize = aSize * (200.0 / -mvPos.z);
  gl_Position = projectionMatrix * mvPos;

  // Fade based on distance from centre — outer particles dimmer
  vAlpha = smoothstep(0.0, 0.3, aRadius) * (0.35 + 0.65 * smoothstep(2.5, 1.0, aRadius));
}
`

// ─── Particle fragment shader — circular soft point ──────────────────────────
const particleFragmentShader = `
uniform vec3 uParticleColor;

varying float vAlpha;

void main() {
  // Circular falloff
  float d = length(gl_PointCoord - vec2(0.5));
  if (d > 0.5) discard;
  float alpha = smoothstep(0.5, 0.15, d) * vAlpha;
  gl_FragColor = vec4(uParticleColor, alpha);
}
`

// Phase colour palettes
const PHASE_COLORS: Record<OrbPhase, {
  core: THREE.Color; mid: THREE.Color; rim: THREE.Color; turbulence: number;
  particle: THREE.Color; particleSpeed: number; particleSpread: number
}> = {
  idle: {
    core: new THREE.Color(0x0d1f3a),
    mid: new THREE.Color(0x1a3a5c),
    rim: new THREE.Color(0x4a9ebe),
    turbulence: 0.15,
    particle: new THREE.Color(0x4a9ebe),
    particleSpeed: 0.3,
    particleSpread: 1.0,
  },
  thinking: {
    core: new THREE.Color(0x0d1a3f),
    mid: new THREE.Color(0x2040a0),
    rim: new THREE.Color(0x60a0ff),
    turbulence: 0.7,
    particle: new THREE.Color(0x60a0ff),
    particleSpeed: 0.8,
    particleSpread: 1.2,
  },
  working: {
    core: new THREE.Color(0x1a0d3f),
    mid: new THREE.Color(0x5020a0),
    rim: new THREE.Color(0xa060ff),
    turbulence: 1.0,
    particle: new THREE.Color(0xc080ff),
    particleSpeed: 1.5,
    particleSpread: 1.5,
  },
  finalizing: {
    core: new THREE.Color(0x0d2e2a),
    mid: new THREE.Color(0x1a6658),
    rim: new THREE.Color(0x4affcc),
    turbulence: 0.3,
    particle: new THREE.Color(0x4affcc),
    particleSpeed: 0.5,
    particleSpread: 1.1,
  },
}

// ─── Create the particle system geometry + material ──────────────────────────
function createParticles(count: number) {
  const geo = new THREE.BufferGeometry()
  const radius = new Float32Array(count)
  const theta  = new Float32Array(count)
  const phi    = new Float32Array(count)
  const speed  = new Float32Array(count)
  const sizes  = new Float32Array(count)

  for (let i = 0; i < count; i++) {
    // Distribute between near-surface (1.15) and outer halo (2.4)
    radius[i] = 1.15 + Math.random() * 1.25
    theta[i]  = Math.random() * Math.PI * 2
    phi[i]    = Math.acos(2 * Math.random() - 1)
    speed[i]  = 0.4 + Math.random() * 0.8
    sizes[i]  = 1.0 + Math.random() * 2.5
  }

  // Need dummy positions — actual pos computed in vertex shader
  const positions = new Float32Array(count * 3)
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geo.setAttribute('aRadius',  new THREE.BufferAttribute(radius, 1))
  geo.setAttribute('aTheta',   new THREE.BufferAttribute(theta, 1))
  geo.setAttribute('aPhi',     new THREE.BufferAttribute(phi, 1))
  geo.setAttribute('aSpeed',   new THREE.BufferAttribute(speed, 1))
  geo.setAttribute('aSize',    new THREE.BufferAttribute(sizes, 1))

  const mat = new THREE.ShaderMaterial({
    vertexShader:   particleVertexShader,
    fragmentShader: particleFragmentShader,
    uniforms: {
      uTime:          { value: 0 },
      uSpeed:         { value: 0.3 },
      uSpread:        { value: 1.0 },
      uParticleColor: { value: PHASE_COLORS.idle.particle.clone() },
    },
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })

  return { geo, mat, points: new THREE.Points(geo, mat) }
}

// ─── Component ───────────────────────────────────────────────────────────────

export default function AgentOrb({ phase, size = 120 }: AgentOrbProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const phaseRef = useRef(phase)
  phaseRef.current = phase

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(size, size)
    renderer.setClearColor(0x000000, 0)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100)
    camera.position.z = 2.8

    // Orb mesh
    const geo = new THREE.SphereGeometry(1, 96, 96)
    const mat = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uTime:      { value: 0 },
        uTurbulence:{ value: PHASE_COLORS.idle.turbulence },
        uColorCore: { value: PHASE_COLORS.idle.core.clone() },
        uColorMid:  { value: PHASE_COLORS.idle.mid.clone() },
        uColorRim:  { value: PHASE_COLORS.idle.rim.clone() },
      },
      transparent: true,
    })
    const mesh = new THREE.Mesh(geo, mat)
    scene.add(mesh)

    // Particle field — scale count with canvas size for perf
    const particleCount = size >= 64 ? 180 : 60
    const particles = createParticles(particleCount)
    scene.add(particles.points)

    // Lerp targets
    let targetTurb = PHASE_COLORS.idle.turbulence
    let targetSpeed = PHASE_COLORS.idle.particleSpeed
    let targetSpread = PHASE_COLORS.idle.particleSpread
    const targetCore = PHASE_COLORS.idle.core.clone()
    const targetMid  = PHASE_COLORS.idle.mid.clone()
    const targetRim  = PHASE_COLORS.idle.rim.clone()
    const targetParticleColor = PHASE_COLORS.idle.particle.clone()

    let rafId: number
    let lastTime = 0

    const animate = (time: number) => {
      rafId = requestAnimationFrame(animate)
      const dt = Math.min((time - lastTime) / 1000, 0.05)
      lastTime = time

      const p = phaseRef.current
      const palette = PHASE_COLORS[p]
      targetTurb   = palette.turbulence
      targetSpeed  = palette.particleSpeed
      targetSpread = palette.particleSpread
      targetCore.set(palette.core)
      targetMid.set(palette.mid)
      targetRim.set(palette.rim)
      targetParticleColor.set(palette.particle)

      const lerpSpeed = dt * 1.8
      const t = time * 0.001

      // Orb uniforms
      mat.uniforms.uTurbulence.value += (targetTurb - mat.uniforms.uTurbulence.value) * lerpSpeed
      ;(mat.uniforms.uColorCore.value as THREE.Color).lerp(targetCore, lerpSpeed)
      ;(mat.uniforms.uColorMid.value as THREE.Color).lerp(targetMid, lerpSpeed)
      ;(mat.uniforms.uColorRim.value as THREE.Color).lerp(targetRim, lerpSpeed)
      mat.uniforms.uTime.value = t

      mesh.rotation.y = time * 0.0003
      mesh.rotation.x = Math.sin(time * 0.0001) * 0.12

      // Particle uniforms — lerp toward target
      const pMat = particles.mat
      pMat.uniforms.uTime.value = t
      pMat.uniforms.uSpeed.value  += (targetSpeed  - pMat.uniforms.uSpeed.value)  * lerpSpeed
      pMat.uniforms.uSpread.value += (targetSpread - pMat.uniforms.uSpread.value) * lerpSpeed
      ;(pMat.uniforms.uParticleColor.value as THREE.Color).lerp(targetParticleColor, lerpSpeed)

      renderer.render(scene, camera)
    }

    rafId = requestAnimationFrame(animate)

    return () => {
      cancelAnimationFrame(rafId)
      geo.dispose()
      mat.dispose()
      particles.geo.dispose()
      particles.mat.dispose()
      renderer.dispose()
    }
  }, [size])

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      style={{ display: 'block', borderRadius: '50%' }}
    />
  )
}
