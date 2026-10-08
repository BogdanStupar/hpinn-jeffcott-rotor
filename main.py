import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import warnings
warnings.filterwarnings("ignore")
from gui import RotorGUI


def main():
    print("=" * 56)
    print("  HPINN Rotor Kit Simulator")
    print("  Bezdimenzionalni Jeffcott rotor model")
    print("=" * 56)
    print("  KORAK 1: Kliknite 1 - Pokreni akviziciju (test-sto)")
    print("  KORAK 2: Kliknite 2 - Nauci ODE parametre")
    print("  KORAK 3: Kliknite 3 - Pokreni dijagnostiku")
    print("  KORAK 4: Pomicite slajdere za injektiranje kvara")
    print("=" * 56)
    print(f"\n  PyTorch: DA  (uredjaj: "
          f"{'CUDA' if torch.cuda.is_available() else 'CPU'})")
    print(f"  Tipovi kvarova: Dеbalans | Nesaosnost | Trenje | Krutost | Prigusenje\n")

    gui = RotorGUI()
    gui.show()


if __name__ == "__main__":
    main()
