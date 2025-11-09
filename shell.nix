{ pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/205fd4226592cc83fd4c0885a3e4c9c400efabb5.tar.gz") { } }:

let
  python = pkgs.python310;  # PyRES requires Python >= 3.10
in
pkgs.mkShell {
  name = "pyres-dev";

  buildInputs = with pkgs; [
    python
    python.pkgs.venvShellHook
    python.pkgs.pip
    python.pkgs.tkinter
    python.pkgs.pyqt6
    python.pkgs.pysoundfile
    stdenv
    gcc
    libsndfile
    xorg.libX11
    zlib
    ffmpeg
  ];

  venvDir = "./.venv";
  postVenvCreation = ''
    echo "Creating virtualenv in $venvDir..."
    pip install --upgrade pip setuptools wheel
    pip install torch torchaudio
    pip install -r requirements.txt
  '';

  postShellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.xorg.libX11}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.ffmpeg.lib}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=${pkgs.zlib}/lib:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    export MPLBACKEND="qtAgg"
    export QT_QPA_PLATFORM=xcb
    echo "Virtualenv is active at $venvDir"
  '';
}
