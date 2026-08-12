{
  description = "animate-any-mesh.cpp — safe GGML CPU/Vulkan inference";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    ggml = {
      url = "github:ggml-org/ggml/331b9cba52b23d895bc4ad218c007eb5e667540f";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, ggml }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in {
      packages = forAllSystems (system:
        let pkgs = import nixpkgs { inherit system; };
        in {
          default = pkgs.stdenv.mkDerivation {
            pname = "animate-any-mesh.cpp";
            version = "0.1.0";
            src = self;
            nativeBuildInputs = with pkgs; [ cmake ninja pkg-config makeWrapper ];
            buildInputs = with pkgs; [
              nlohmann_json
              meshoptimizer
              utf8proc
              shaderc
              vulkan-headers
              vulkan-loader
            ];
            cmakeFlags = [
              "-DAAM_GGML_SOURCE_DIR=${ggml}"
              "-DAAM_VULKAN=ON"
              "-DAAM_DYNAMIC_BACKENDS=ON"
              "-DAAM_CPU_ALL_VARIANTS=ON"
              "-DAAM_BUILD_TESTS=ON"
            ];
            doCheck = true;
            checkPhase = "ctest --output-on-failure";
            postFixup = ''
              wrapProgram "$out/bin/aam-cli" \
                --prefix LD_LIBRARY_PATH : "${pkgs.vulkan-loader}/lib:/run/opengl-driver/lib"
            '';
          };

          demo = pkgs.buildGoModule {
            pname = "animate-any-mesh-demo";
            version = "0.1.0";
            src = self;
            modRoot = "demo";
            vendorHash = null;
            subPackages = [ "." ];
          };
        });

      apps = forAllSystems (system: {
        demo = {
          type = "app";
          program = "${self.packages.${system}.demo}/bin/animate-any-mesh-demo";
        };
      });

      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          py = pkgs.python3.withPackages (ps: with ps; [ numpy ]);
          common = with pkgs; [
            cmake
            ninja
            pkg-config
            git
            gdb
            nlohmann_json
            meshoptimizer
            utf8proc
            shaderc
            spirv-tools
            vulkan-headers
            vulkan-loader
            vulkan-tools
            go
            py
          ];
          hook = ''
            export AAM_GGML_SOURCE_DIR=${ggml}
            export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib:/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';
        in {
          default = pkgs.mkShell {
            packages = common ++ [ pkgs.gcc ];
            shellHook = hook;
          };
          fuzz = pkgs.mkShell {
            packages = common ++ [ pkgs.clang pkgs.llvm ];
            shellHook = hook;
          };
        });
    };
}
