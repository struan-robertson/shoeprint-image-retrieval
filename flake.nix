{
  # Bit of an ugly impure solution but only PyTorch 1.13.1 works with my GPU so use pip to install it, other dependencies are installed through
  description = "PyTorch 1.13.1 Development Environment";

  inputs = { nixpkgs.url = "github:nixos/nixpkgs/nixos-23.05"; };

  outputs = { self, nixpkgs, ... }:
    let
      # system should match the system you are running on
      system = "x86_64-linux";
    in {
      devShells."${system}".default = let

        pkgs = import nixpkgs {
          inherit system;
          overlays = [ ];
        };

      in pkgs.mkShell rec {
        # create an environment with python 3.10

        name = "impurePythonEnv";
        venvDir = "./.venv";

        buildInputs = with pkgs;
          [
            # A Python interpreter including the 'venv' module is required to bootstrap
            # the environment.
            python310

            # In this particular example, in order to compile any binary extensions they may
            # require, the Python modules listed in the hypothetical requirements.txt need
            # the following packages to be installed locally:
            libdrm
          ] ++ (with pkgs.python310Packages; [
            # This executes some shell code to initialize a venv in $venvDir before
            # dropping into the shell
            venvShellHook

            # Normal requirements
            opencv4
            keras
            numba
            pillow
            numpy
            scipy
            tensorflow
            tqdm
            joblib
            matplotlib
            timm

            # Dev requirements
            ipython
            ipdb
            pyqt6
            rich
            line_profiler
            torchinfo

          ]);

        # Run this command, only after creating the virtual environment
        postVenvCreation = ''
          unset SOURCE_DATE_EPOCH
          pip install -r requirements.txt
        '';

        postShellHook = ''
          # allow pip to install wheels
          unset SOURCE_DATE_EPOCH
          # Fix missing libstd
          export LD_LIBRARY_PATH=${
            pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]
          }
        '';

        # Set matplotlib backend
        MPLBACKEND = "QtAGG";

        # Use ROCM with radeon rx 6750 XT
        HSA_OVERRIDE_GFX_VERSION = "10.3.0";

      };
    };
}
