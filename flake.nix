{
  description = "ParaView + Meshoid dev shell for a GIZMO (meshless) data plugin";

  inputs = {
    # Pinned to a rev whose paraview/vtk are in cache.nixos.org (no local VTK build).
    nixpkgs.url = "github:nixos/nixpkgs/151fa4e8ddfdd8dd25d945ad94ed54a13de9f6e4";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };

        # ParaView's embedded interpreter is VTK's python. Everything that
        # pvpython / the GUI imports must be built against this exact python.
        py = pkgs.vtk-full.vtkPackages.python3Packages;

        # meshoid itself is used from the local clone ./meshoid (editable-style,
        # via PYTHONPATH in shellHook); nix only supplies its dependencies.
        pythonEnv = py.python.withPackages (ps: with ps; [
          numpy scipy numba h5py astropy matplotlib
          ipython pytest
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            paraview
            pythonEnv
            # C++ plugin toolchain (ParaView's CMake config propagates qt6/vtk)
            cmake ninja gcc gdb pkg-config
            qt6Packages.wrapQtAppsHook
            hdf5
            git
            xvfb-run   # headless GUI tests: xvfb-run -a paraview --script=... --exit
          ];

          shellHook = ''
            export PV_GIZMO_ROOT=$PWD
            # meshoid is a git-ignored local clone (editable). Recreate it if missing.
            if [ ! -d "$PWD/meshoid" ]; then
              git clone -q https://github.com/mikegrudic/meshoid "$PWD/meshoid" \
                && git -C "$PWD/meshoid" checkout -q ad32966
            fi
            # sys.executable is pvpython/paraview, which can't run multiprocessing
            # helpers (numba's parallel init spawns resource_tracker). Plugins call
            # multiprocessing.set_executable($PV_GIZMO_PYTHON) before importing numba.
            export PV_GIZMO_PYTHON=${pythonEnv}/bin/python3
            # Make pythonEnv's site-packages (meshoid, numba, h5py...) visible
            # inside paraview / pvpython, which use VTK's bare interpreter.
            export PYTHONPATH=$PWD/meshoid/src:${pythonEnv}/${pythonEnv.sitePackages}''${PYTHONPATH:+:$PYTHONPATH}
            # Auto-load our Python plugin(s) when present.
            if [ -d "$PWD/plugin" ]; then
              export PV_PLUGIN_PATH=$PWD/plugin''${PV_PLUGIN_PATH:+:$PV_PLUGIN_PATH}
            fi
            export NUMBA_CACHE_DIR=''${XDG_CACHE_HOME:-$HOME/.cache}/numba
            echo "ParaView $(pvpython --version 2>/dev/null | tail -1) | meshoid dev shell"
          '';
        };
      });
}
