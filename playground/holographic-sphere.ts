// Holographic sphere effect for the Babylon.js TypeScript Playground.
//
// Pipeline:
//   1. baseDepthRTT   -> linear view-space depth of the scene WITHOUT the holo sphere
//                        (ground + probe), used to occlude the hologram correctly.
//   2. sphereBackRTT  -> holo sphere, front faces culled (only the far shell), color + depth.
//   3. sphereFrontRTT -> holo sphere, back faces culled (only the near shell), color + depth.
//   4. holoComposite  -> depth-tests both shells against the scene depth and blends the
//                        back shell at 0.25 and the front shell at 0.65 over the scene.
//
// The holo sphere lives on its own layer + camera so the main pass never draws it and the
// offscreen sphere passes render it reliably. All depth is stored as linear view depth
// (viewZ / far) with an identical formula in every pass, so comparisons are reverse-Z safe.

class Playground {
    public static CreateScene(engine: BABYLON.Engine, canvas: HTMLCanvasElement): BABYLON.Scene {
        const scene = new BABYLON.Scene(engine);
        scene.clearColor = new BABYLON.Color4(0.02, 0.03, 0.06, 1.0);

        const camera = new BABYLON.FreeCamera("camera1", new BABYLON.Vector3(0, 5, -10), scene);
        camera.setTarget(BABYLON.Vector3.Zero());
        camera.attachControl(canvas, true);
        camera.minZ = 0.1;
        camera.maxZ = 100.0;

        const light = new BABYLON.HemisphericLight("light", new BABYLON.Vector3(0, 1, 0), scene);
        light.intensity = 0.7;

        // The holographic subject.
        const sphere = BABYLON.MeshBuilder.CreateSphere("sphere", { diameter: 2, segments: 48 }, scene);
        sphere.position.y = 1;

        // A dark ground so the hologram reads against the scene.
        const ground = BABYLON.MeshBuilder.CreateGround("ground", { width: 12, height: 12 }, scene);
        const groundMat = new BABYLON.StandardMaterial("groundMat", scene);
        groundMat.diffuseColor = new BABYLON.Color3(0.03, 0.05, 0.09);
        groundMat.specularColor = new BABYLON.Color3(0.05, 0.08, 0.12);
        ground.material = groundMat;

        // Solid probe that orbits through the hologram to prove depth-correct occlusion.
        const probe = BABYLON.MeshBuilder.CreateSphere("probe", { diameter: 0.9, segments: 24 }, scene);
        const probeMat = new BABYLON.StandardMaterial("probeMat", scene);
        probeMat.diffuseColor = new BABYLON.Color3(1.0, 0.55, 0.15);
        probeMat.emissiveColor = new BABYLON.Color3(0.2, 0.08, 0.0);
        probe.material = probeMat;
        probe.position.set(-3, 1, 0);

        // ------------------------------------------------------------------ config
        const HoloLayer = 0x10000000;
        const MainLayer = 0x0fffffff;

        const LineColor = new BABYLON.Vector3(0.35, 0.95, 1.0);
        const BaseColorFront = new BABYLON.Vector3(0.0, 0.5, 0.8);
        const BaseColorBack = new BABYLON.Vector3(0.0, 0.22, 0.42);
        const LineFrequency = 22.0;
        const LineSpeed = 2.4;
        const BackOpacity = 0.25;
        const FrontOpacity = 0.65;
        const InvFar = 1.0 / camera.maxZ;

        // Keep the holo sphere off the main camera; render it only through a matching helper camera.
        sphere.layerMask = HoloLayer;
        camera.layerMask = MainLayer;

        const holoCamera = new BABYLON.FreeCamera("holoCamera", camera.position.clone(), scene);
        holoCamera.layerMask = HoloLayer;
        holoCamera.minZ = camera.minZ;
        holoCamera.maxZ = camera.maxZ;
        holoCamera.fov = camera.fov;
        scene.activeCamera = camera;

        // ------------------------------------------------------------------ shaders
        const depthVertex = `
            precision highp float;
            attribute vec3 position;
            uniform mat4 worldView;
            uniform mat4 viewProjection;
            uniform mat4 world;
            uniform float invFar;
            varying float vLinearDepth;
            void main(void) {
                vec4 viewPos = worldView * vec4(position, 1.0);
                vLinearDepth = clamp(-viewPos.z * invFar, 0.0, 1.0);
                gl_Position = viewProjection * world * vec4(position, 1.0);
            }
        `;
        const depthFragment = `
            precision highp float;
            varying float vLinearDepth;
            void main(void) {
                gl_FragColor = vec4(vLinearDepth, 0.0, 0.0, 1.0);
            }
        `;

        const holoVertex = `
            precision highp float;
            attribute vec3 position;
            attribute vec3 normal;
            uniform mat4 world;
            uniform mat4 worldView;
            uniform mat4 viewProjection;
            uniform float invFar;
            varying vec3 vPositionW;
            varying vec3 vNormalW;
            varying float vLinearDepth;
            void main(void) {
                vec4 worldPos = world * vec4(position, 1.0);
                vPositionW = worldPos.xyz;
                vNormalW = normalize((world * vec4(normal, 0.0)).xyz);
                vec4 viewPos = worldView * vec4(position, 1.0);
                vLinearDepth = clamp(-viewPos.z * invFar, 0.0, 1.0);
                gl_Position = viewProjection * worldPos;
            }
        `;
        const holoFragment = `
            precision highp float;
            varying vec3 vPositionW;
            varying vec3 vNormalW;
            varying float vLinearDepth;
            uniform vec3 cameraPositionW;
            uniform vec3 baseColor;
            uniform vec3 lineColor;
            uniform float time;
            uniform float lineFrequency;
            uniform float lineSpeed;
            uniform float faceGlow;

            float hash(vec2 p) {
                return fract(sin(dot(p, vec2(41.3, 289.1))) * 43758.5453);
            }

            void main(void) {
                vec3 N = normalize(vNormalW);
                vec3 V = normalize(cameraPositionW - vPositionW);
                float fresnel = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 3.0);

                // Vertical scan lines that ripple sinusoidally along Y and drift over time.
                float ripple = sin(vPositionW.y * 3.0 + time * 1.5) * 0.15;
                float coord = (vPositionW.x + ripple) * lineFrequency - time * lineSpeed;
                float lines = smoothstep(0.2, 0.95, sin(coord));

                // Travelling refresh bar sweeping down the model.
                float bar = fract(vPositionW.y * 0.2 - time * 0.12);
                float scanBar = smoothstep(0.0, 0.03, bar) * (1.0 - smoothstep(0.03, 0.09, bar));

                // Digital flicker.
                float flick = 0.85 + 0.15 * hash(vec2(floor(time * 20.0), floor(vPositionW.y * 8.0)));

                vec3 col = baseColor;
                col += lineColor * lines * 1.4;
                col += lineColor * fresnel * 2.0;
                col += lineColor * scanBar * 1.2;
                col *= flick;
                col += baseColor * faceGlow;

                // Alpha carries linear view depth for the composite occlusion test.
                gl_FragColor = vec4(col, vLinearDepth);
            }
        `;

        // ------------------------------------------------------------------ base depth pass
        const depthMat = new BABYLON.ShaderMaterial("holoDepthMat", scene, {
            vertexSource: depthVertex,
            fragmentSource: depthFragment
        }, {
            attributes: ["position"],
            uniforms: ["world", "worldView", "viewProjection", "invFar"]
        });
        depthMat.setFloat("invFar", InvFar);

        const baseDepthRTT = new BABYLON.RenderTargetTexture(
            "baseDepthRTT",
            { width: engine.getRenderWidth(), height: engine.getRenderHeight() },
            scene, false, true, BABYLON.Constants.TEXTURETYPE_FLOAT
        );
        baseDepthRTT.clearColor = new BABYLON.Color4(1, 0, 0, 1); // far
        baseDepthRTT.activeCamera = camera;
        baseDepthRTT.renderList = [ground, probe];

        const savedMaterials = new Map<BABYLON.AbstractMesh, BABYLON.Nullable<BABYLON.Material>>();
        baseDepthRTT.onBeforeRenderObservable.add(() => {
            savedMaterials.clear();
            baseDepthRTT.renderList!.forEach((m) => {
                savedMaterials.set(m, m.material);
                m.material = depthMat;
            });
        });
        baseDepthRTT.onAfterRenderObservable.add(() => {
            baseDepthRTT.renderList!.forEach((m) => {
                m.material = savedMaterials.get(m) ?? null;
            });
        });
        scene.customRenderTargets.push(baseDepthRTT);

        // ------------------------------------------------------------------ holo sphere passes
        const makeHoloMaterial = (name: string, cull: "front" | "back", baseColor: BABYLON.Vector3, glow: number): BABYLON.ShaderMaterial => {
            const mat = new BABYLON.ShaderMaterial(name, scene, {
                vertexSource: holoVertex,
                fragmentSource: holoFragment
            }, {
                attributes: ["position", "normal"],
                uniforms: ["world", "worldView", "viewProjection", "invFar", "cameraPositionW", "baseColor", "lineColor", "time", "lineFrequency", "lineSpeed", "faceGlow"]
            });
            mat.backFaceCulling = true;
            // Flipping the winding interpretation flips which faces survive culling, so the
            // "back" material keeps the far shell and the "front" material keeps the near shell.
            mat.sideOrientation = cull === "back"
                ? BABYLON.Material.ClockWiseSideOrientation
                : BABYLON.Material.CounterClockWiseSideOrientation;
            mat.setFloat("invFar", InvFar);
            mat.setVector3("baseColor", baseColor);
            mat.setVector3("lineColor", LineColor);
            mat.setFloat("lineFrequency", LineFrequency);
            mat.setFloat("lineSpeed", LineSpeed);
            mat.setFloat("faceGlow", glow);
            mat.setFloat("time", 0);
            mat.setVector3("cameraPositionW", camera.globalPosition);
            return mat;
        };

        const backMat = makeHoloMaterial("holoBackMat", "back", BaseColorBack, 0.0);
        const frontMat = makeHoloMaterial("holoFrontMat", "front", BaseColorFront, 0.15);

        const makeSphereRTT = (name: string, material: BABYLON.ShaderMaterial): BABYLON.RenderTargetTexture => {
            const rtt = new BABYLON.RenderTargetTexture(
                name,
                { width: engine.getRenderWidth(), height: engine.getRenderHeight() },
                scene, false, true, BABYLON.Constants.TEXTURETYPE_FLOAT
            );
            rtt.clearColor = new BABYLON.Color4(0, 0, 0, 1); // rgb empty, alpha = far
            rtt.activeCamera = holoCamera;
            rtt.renderList = [sphere];
            rtt.onBeforeRenderObservable.add(() => { sphere.material = material; });
            scene.customRenderTargets.push(rtt);
            return rtt;
        };

        const sphereBackRTT = makeSphereRTT("sphereBackRTT", backMat);
        const sphereFrontRTT = makeSphereRTT("sphereFrontRTT", frontMat);
        sphereFrontRTT.onAfterRenderObservable.add(() => { sphere.material = null; });

        // ------------------------------------------------------------------ composite
        BABYLON.Effect.ShadersStore["holoCompositeFragmentShader"] = `
            precision highp float;
            varying vec2 vUV;
            uniform sampler2D textureSampler;
            uniform sampler2D sceneDepthSampler;
            uniform sampler2D sphereFrontSampler;
            uniform sampler2D sphereBackSampler;
            uniform float backOpacity;
            uniform float frontOpacity;
            uniform float time;

            void main(void) {
                vec3 sceneColor = texture2D(textureSampler, vUV).rgb;
                float sceneDepth = texture2D(sceneDepthSampler, vUV).r;

                vec4 back = texture2D(sphereBackSampler, vUV);

                // Slight chromatic split on the near shell for a holographic shimmer.
                float ca = 0.0015 + 0.001 * sin(time * 3.0 + vUV.y * 40.0);
                vec4 front = texture2D(sphereFrontSampler, vUV);
                vec3 frontRGB = vec3(
                    texture2D(sphereFrontSampler, vUV + vec2(ca, 0.0)).r,
                    front.g,
                    texture2D(sphereFrontSampler, vUV - vec2(ca, 0.0)).b
                );

                vec3 result = sceneColor;
                float bias = 0.0008;

                // Far shell first (dim), then near shell (bright), each occluded by real geometry.
                if (back.a < 0.9995 && back.a <= sceneDepth + bias) {
                    result = mix(result, back.rgb, backOpacity);
                }
                if (front.a < 0.9995 && front.a <= sceneDepth + bias) {
                    result = mix(result, frontRGB, frontOpacity);
                }

                gl_FragColor = vec4(result, 1.0);
            }
        `;

        const composite = new BABYLON.PostProcess(
            "holoComposite",
            "holoComposite",
            ["backOpacity", "frontOpacity", "time"],
            ["sceneDepthSampler", "sphereFrontSampler", "sphereBackSampler"],
            1.0,
            camera
        );

        let elapsed = 0;
        composite.onApply = (effect: BABYLON.Effect) => {
            effect.setTexture("sceneDepthSampler", baseDepthRTT);
            effect.setTexture("sphereFrontSampler", sphereFrontRTT);
            effect.setTexture("sphereBackSampler", sphereBackRTT);
            effect.setFloat("backOpacity", BackOpacity);
            effect.setFloat("frontOpacity", FrontOpacity);
            effect.setFloat("time", elapsed);
        };

        // ------------------------------------------------------------------ per-frame
        scene.onBeforeRenderObservable.add(() => {
            elapsed += engine.getDeltaTime() * 0.001;

            // Keep the helper camera locked to the main view so the overlay aligns.
            holoCamera.position.copyFrom(camera.position);
            holoCamera.rotation.copyFrom(camera.rotation);
            holoCamera.fov = camera.fov;

            frontMat.setFloat("time", elapsed);
            backMat.setFloat("time", elapsed);
            frontMat.setVector3("cameraPositionW", camera.globalPosition);
            backMat.setVector3("cameraPositionW", camera.globalPosition);

            probe.position.x = Math.sin(elapsed * 0.6) * 3.0;
        });

        engine.onResizeObservable.add(() => {
            const w = engine.getRenderWidth();
            const h = engine.getRenderHeight();
            baseDepthRTT.resize({ width: w, height: h });
            sphereBackRTT.resize({ width: w, height: h });
            sphereFrontRTT.resize({ width: w, height: h });
        });

        // ------------------------------------------------------------------ bloom
        const pipeline = new BABYLON.DefaultRenderingPipeline("holoPipeline", true, scene, [camera]);
        pipeline.bloomEnabled = true;
        pipeline.bloomThreshold = 0.35;
        pipeline.bloomWeight = 0.6;
        pipeline.bloomKernel = 64;
        pipeline.bloomScale = 0.5;

        return scene;
    }
}
export { Playground };
