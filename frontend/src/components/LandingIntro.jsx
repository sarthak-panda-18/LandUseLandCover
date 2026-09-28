import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { AlertCircle, Satellite, Info, ShieldCheck, Sparkles, Layers } from 'lucide-react';

export default function LandingIntro() {
  const mountRef = useRef(null);
  const [webglSupported, setWebglSupported] = useState(true);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    let scene, camera, renderer, animationFrameId;
    let controls, resumeTimeout;
    let worldGroup, satellitePivot;
    let resizeObserver;

    try {
      // 1. Scene Setup
      scene = new THREE.Scene();

      const width = container.clientWidth || 320;
      const height = container.clientHeight || 280;

      // 2. Camera
      camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
      camera.position.set(0, 1.2, 7.8);
      camera.lookAt(0, 0, 0);

      // 3. Renderer
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: 'low-power',
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      renderer.setClearColor(0x000000, 0); // Transparent background
      renderer.domElement.style.touchAction = 'pan-y';
      renderer.domElement.style.cursor = 'grab';
      container.appendChild(renderer.domElement);

      // 4. Interactive OrbitControls (Rotate-only, no zoom or pan to allow natural page scrolling)
      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.enableZoom = false; // Disable zooming so page scrolling is uninterrupted
      controls.enablePan = false;  // Keep rotate-only
      controls.autoRotate = true;  // Slow auto-rotation
      controls.autoRotateSpeed = 1.0;

      controls.addEventListener('start', () => {
        if (resumeTimeout) clearTimeout(resumeTimeout);
        controls.autoRotate = false;
        if (renderer.domElement) renderer.domElement.style.cursor = 'grabbing';
      });

      controls.addEventListener('end', () => {
        if (renderer.domElement) renderer.domElement.style.cursor = 'grab';
        if (resumeTimeout) clearTimeout(resumeTimeout);
        // Resume auto-rotation 2 seconds after user stops dragging
        resumeTimeout = setTimeout(() => {
          controls.autoRotate = true;
        }, 2000);
      });

      // 5. Lighting (Warm ambient + directional highlights)
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
      scene.add(ambientLight);

      const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.4);
      dirLight1.position.set(6, 8, 6);
      scene.add(dirLight1);

      const dirLight2 = new THREE.DirectionalLight(0x2E86AB, 0.7); // Sky blue secondary rim
      dirLight2.position.set(-6, -4, -4);
      scene.add(dirLight2);

      // 6. Stylized Low-Poly Earth Globe & Satellite System
      worldGroup = new THREE.Group();
      scene.add(worldGroup);

      // (a) Base Ocean Icosahedron Globe
      const globeGeo = new THREE.IcosahedronGeometry(2.1, 2);
      const globeMat = new THREE.MeshStandardMaterial({
        color: 0x0B3954, // Primary navy
        flatShading: true,
        roughness: 0.55,
        metalness: 0.15,
      });
      const globeMesh = new THREE.Mesh(globeGeo, globeMat);
      worldGroup.add(globeMesh);

      // (b) Subtle Wireframe Grid Overlay
      const wireGeo = new THREE.IcosahedronGeometry(2.13, 2);
      const wireMat = new THREE.MeshBasicMaterial({
        color: 0x2E86AB, // Secondary sky blue
        wireframe: true,
        transparent: true,
        opacity: 0.35,
      });
      const wireMesh = new THREE.Mesh(wireGeo, wireMat);
      worldGroup.add(wireMesh);

      // (c) Low-Poly Land Mass Clusters (Forest Green & Sand)
      const landCoords = [
        { x: 1.3, y: 1.2, z: 1.0, scale: 0.65, color: 0x2D6A4F },
        { x: -1.4, y: 0.8, z: 1.1, scale: 0.55, color: 0x2D6A4F },
        { x: 0.2, y: -1.5, z: 1.3, scale: 0.6, color: 0xD8C3A5 },
        { x: 1.5, y: -0.6, z: -1.0, scale: 0.5, color: 0x2D6A4F },
        { x: -1.2, y: -1.1, z: -1.2, scale: 0.7, color: 0x2D6A4F },
        { x: -0.4, y: 1.6, z: -1.1, scale: 0.55, color: 0xD8C3A5 },
      ];

      landCoords.forEach(({ x, y, z, scale, color }) => {
        const patchGeo = new THREE.DodecahedronGeometry(scale, 0);
        const patchMat = new THREE.MeshStandardMaterial({
          color: color,
          flatShading: true,
          roughness: 0.7,
        });
        const patchMesh = new THREE.Mesh(patchGeo, patchMat);
        patchMesh.position.set(x, y, z);
        worldGroup.add(patchMesh);
      });

      // (d) Orbit Path Ring
      const orbitGeo = new THREE.TorusGeometry(3.5, 0.02, 8, 80);
      const orbitMat = new THREE.MeshBasicMaterial({
        color: 0x2E86AB,
        transparent: true,
        opacity: 0.4,
      });
      const orbitRing = new THREE.Mesh(orbitGeo, orbitMat);
      orbitRing.rotation.x = Math.PI / 2.6;
      orbitRing.rotation.y = Math.PI / 7;
      scene.add(orbitRing);

      // (e) Stylized Sentinel-2 Satellite
      satellitePivot = new THREE.Group();
      satellitePivot.rotation.x = Math.PI / 2.6;
      satellitePivot.rotation.y = Math.PI / 7;
      scene.add(satellitePivot);

      const satModel = new THREE.Group();
      satModel.position.set(3.5, 0, 0);

      // Satellite main body (Gold foil / sand)
      const bodyGeo = new THREE.BoxGeometry(0.35, 0.25, 0.3);
      const bodyMat = new THREE.MeshStandardMaterial({
        color: 0xD8C3A5, // Sand accent
        roughness: 0.3,
        metalness: 0.6,
      });
      const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
      satModel.add(bodyMesh);

      // Solar Panels (Blue wings)
      const panelGeo = new THREE.BoxGeometry(1.2, 0.03, 0.35);
      const panelMat = new THREE.MeshStandardMaterial({
        color: 0x2E86AB, // Secondary sky blue
        roughness: 0.2,
        metalness: 0.85,
      });
      const panelMesh = new THREE.Mesh(panelGeo, panelMat);
      satModel.add(panelMesh);

      // Multispectral Imager / Sensor aperture (Emerald green)
      const sensorGeo = new THREE.CylinderGeometry(0.07, 0.09, 0.12, 12);
      const sensorMat = new THREE.MeshStandardMaterial({
        color: 0x06A77D, // Success emerald
        roughness: 0.2,
      });
      const sensorMesh = new THREE.Mesh(sensorGeo, sensorMat);
      sensorMesh.rotation.x = Math.PI / 2;
      sensorMesh.position.set(0, -0.15, 0);
      satModel.add(sensorMesh);

      satellitePivot.add(satModel);

      // 7. Smooth Animation Loop
      let prevTime = performance.now();
      const animate = (currentTime) => {
        animationFrameId = requestAnimationFrame(animate);
        const delta = Math.min((currentTime - prevTime) / 1000, 0.1);
        prevTime = currentTime;

        // Update OrbitControls (handles smooth damping & autoRotate)
        controls.update();

        // Satellite orbital rotation
        satellitePivot.rotation.z += delta * 0.45;
        satModel.rotation.y += delta * 0.8;

        renderer.render(scene, camera);
      };

      animate(performance.now());

      // 8. Responsive Resizing
      const handleResize = () => {
        if (!container || !renderer || !camera) return;
        const newW = container.clientWidth;
        const newH = container.clientHeight;
        if (newW > 0 && newH > 0) {
          camera.aspect = newW / newH;
          camera.updateProjectionMatrix();
          renderer.setSize(newW, newH);
        }
      };

      if (window.ResizeObserver) {
        resizeObserver = new ResizeObserver(() => handleResize());
        resizeObserver.observe(container);
      } else {
        window.addEventListener('resize', handleResize);
      }
    } catch (err) {
      console.warn('Three.js WebGL initialization failed or unsupported:', err);
      setWebglSupported(false);
    }

    // 9. Cleanup & Disposal on Unmount
    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (resumeTimeout) clearTimeout(resumeTimeout);
      if (controls) controls.dispose();
      if (resizeObserver) resizeObserver.disconnect();
      else window.removeEventListener('resize', handleResize);

      if (scene) {
        scene.traverse((object) => {
          if (object.geometry) object.geometry.dispose();
          if (object.material) {
            if (Array.isArray(object.material)) {
              object.material.forEach((mat) => mat.dispose());
            } else {
              object.material.dispose();
            }
          }
        });
      }

      if (renderer) {
        renderer.dispose();
        if (renderer.domElement && container.contains(renderer.domElement)) {
          container.removeChild(renderer.domElement);
        }
      }
    };
  }, []);

  return (
    <section className="landing-intro-hero" aria-label="Introduction and Requirements">
      <div className="container landing-intro-container">
        {/* Left / Main Content Column */}
        <div className="landing-intro-content">
          <div className="hero-eyebrow">
            <span className="eyebrow-dot"></span>
            <Satellite size={14} className="eyebrow-icon" />
            <span>Earth Observation AI • Sentinel-2 Multispectral</span>
          </div>

          <h1 className="hero-title">
            LULC Classification from <span className="title-highlight">Sentinel-2</span> Imagery
          </h1>

          <p className="hero-description">
            High-resolution Land Use & Land Cover mapping powered by an offline-trained 
            Random Forest classifier with automated 10m spectral indices extraction.
          </p>

          {/* Prominent TIFF-Only Notice Box */}
          <div className="tiff-requirement-card" role="alert">
            <div className="requirement-icon-box">
              <AlertCircle size={22} className="req-icon" />
            </div>
            <div className="requirement-body">
              <div className="requirement-heading">Important Input File Requirement:</div>
              <p className="requirement-text">
                This tool <strong>only accepts GeoTIFF (.tif / .tiff)</strong> files containing 
                <strong> 5-band Sentinel-2 multispectral imagery</strong> (Blue, Green, Red, NIR, NDVI). 
                Regular photos (<code>.png</code>, <code>.jpg</code>, <code>.jpeg</code>) are <strong>not supported</strong>.
              </p>
            </div>
          </div>

          {/* Quick Spec Pills */}
          <div className="hero-spec-pills">
            <div className="spec-pill">
              <Layers size={13} className="spec-icon" />
              <span>5 Spectral Bands (B2, B3, B4, B8, NDVI)</span>
            </div>
            <div className="spec-pill">
              <ShieldCheck size={13} className="spec-icon" />
              <span>10m Spatial Resolution</span>
            </div>
            <div className="spec-pill">
              <Sparkles size={13} className="spec-icon" />
              <span>6 Land Cover Classes</span>
            </div>
          </div>
        </div>

        {/* Right / 3D Canvas Visual Column */}
        <div className="landing-intro-visual">
          <div className="visual-canvas-card">
            <div
              ref={mountRef}
              className="three-canvas-wrapper"
              aria-label="Interactive 3D low-poly Sentinel-2 Earth orbit visualization"
            >
              {!webglSupported && (
                <div className="webgl-fallback">
                  <Satellite size={64} className="fallback-icon" />
                  <span>Sentinel-2 Satellite Orbit</span>
                </div>
              )}
            </div>
            <div className="visual-caption">
              <span className="live-indicator"></span>
              <span>Sentinel-2 10m Orbital AI Processing</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
