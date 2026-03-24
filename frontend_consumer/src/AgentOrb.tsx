import { useEffect, useRef } from 'react'
import * as THREE from 'three'

type OrbPhase = 'idle' | 'thinking' | 'working' | 'finalizing'

interface AgentOrbProps {
  phase: OrbPhase
  size?: number
}

// Vertex shader — displaces sphere vertices with time-driven noise
const vertexShader = `
uniform float uTime;
uniform float uTurbulence;

varying vec3 vNormal;
varying vec3 vPosition;
varying float vDisplacement;

// Simple 3D value noise
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

// Fragment shader — phase-aware color field with fresnel rim
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

  // Subtle shimmer
  float shimmer = sin(vPosition.y * 12.0 + uTime * 1.8) * 0.04;
  col += shimmer;

  float alpha = 0.92 + fresnel * 0.08;
  gl_FragColor = vec4(col, alpha);
}
`

// Phase colour palettes
const PHASE_COLORS: Record<OrbPhase, { core: THREE.Color; mid: THREE.Color; rim: THREE.Color; turbulence: number }> = {
  idle: {
    core: new THREE.Color(0x0d1f3a),
    mid: new THREE.Color(0x1a3a5c),
    rim: new THREE.Color(0x4a9ebe),
    turbulence: 0.15,
  },
  thinking: {
    core: new THREE.Color(0x0d1a3f),
    mid: new THREE.Color(0x2040a0),
    rim: new THREE.Color(0x60a0ff),
    turbulence: 0.7,
  },
  working: {
    core: new THREE.Color(0x1a0d3f),
    mid: new THREE.Color(0x5020a0),
    rim: new THREE.Color(0xa060ff),
    turbulence: 1.0,
  },
  finalizing: {
    core: new THREE.Color(0x0d2e2a),
    mid: new THREE.Color(0x1a6658),
    rim: new THREE.Color(0x4affcc),
    turbulence: 0.3,
  },
}

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

    const geo = new THREE.SphereGeometry(1, 96, 96)
    const mat = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uTurbulence: { value: PHASE_COLORS.idle.turbulence },
        uColorCore: { value: PHASE_COLORS.idle.core.clone() },
        uColorMid: { value: PHASE_COLORS.idle.mid.clone() },
        uColorRim: { value: PHASE_COLORS.idle.rim.clone() },
      },
      transparent: true,
    })

    const mesh = new THREE.Mesh(geo, mat)
    scene.add(mesh)

    // Lerp targets
    let targetTurb = PHASE_COLORS.idle.turbulence
    const targetCore = PHASE_COLORS.idle.core.clone()
    const targetMid = PHASE_COLORS.idle.mid.clone()
    const targetRim = PHASE_COLORS.idle.rim.clone()

    let rafId: number
    let lastTime = 0

    const animate = (time: number) => {
      rafId = requestAnimationFrame(animate)
      const dt = Math.min((time - lastTime) / 1000, 0.05)
      lastTime = time

      const p = phaseRef.current
      const palette = PHASE_COLORS[p]
      targetTurb = palette.turbulence
      targetCore.set(palette.core)
      targetMid.set(palette.mid)
      targetRim.set(palette.rim)

      const lerpSpeed = dt * 1.8
      mat.uniforms.uTurbulence.value += (targetTurb - mat.uniforms.uTurbulence.value) * lerpSpeed
      ;(mat.uniforms.uColorCore.value as THREE.Color).lerp(targetCore, lerpSpeed)
      ;(mat.uniforms.uColorMid.value as THREE.Color).lerp(targetMid, lerpSpeed)
      ;(mat.uniforms.uColorRim.value as THREE.Color).lerp(targetRim, lerpSpeed)

      mat.uniforms.uTime.value = time * 0.001
      mesh.rotation.y = time * 0.0003
      mesh.rotation.x = Math.sin(time * 0.0001) * 0.12

      renderer.render(scene, camera)
    }

    rafId = requestAnimationFrame(animate)

    return () => {
      cancelAnimationFrame(rafId)
      geo.dispose()
      mat.dispose()
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
