{
  description = "PyTorch Development Environment";

  inputs = { nixpkgs.url = "github:nixos/nixpkgs/nixos-23.05"; };

  outputs = { self, nixpkgs, nixpkgs-pytorch, ... }:
    let
      # system should match the system you are running on
      system = "x86_64-linux";
    in {
      devShells."${system}".default = let

        pkgs = import nixpkgs {
          inherit system;
          overlays = [ ];
        };

      in pkgs.mkShell {
        # create an environment with python 3.10

        packages = with pkgs; [ python310 pipenv stdenv.cc.cc.lib ];

        # shellHook = ''
        #   exec fish
        # '';
      };
    };
}
