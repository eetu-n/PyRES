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
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
  '';

  postShellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.libsndfile.out}/lib:${pkgs.xorg.libX11}/lib:${pkgs.zlib}/lib:${pkgs.ffmpeg.lib}/lib:$LD_LIBRARY_PATH
    export MPLBACKEND="qtAgg"
    echo "Virtualenv is active at $venvDir"
  '';
}
