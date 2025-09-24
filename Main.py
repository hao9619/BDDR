import sys


from BDI import bddrmain


from Split_image import simain

from config import Config
from EvaUnite import EUmain

def main():
    # 检查命令行参数数量
    if len(sys.argv) < 2:
        print("Usage: python Main.py <command>")
        print("Available commands:")
        print("  bd - Execute backdoor related functionality")
        sys.exit(1)

    # 获取命令行参数
    command = sys.argv[1]

    opt=Config()

    if command == "bddr":
        bddrmain(opt)
    elif command == "eu":
        EUmain(opt)

    elif command == "si":
        simain()
  
    else:
        print(f"Unknown command: {command}")
        print("Available commands:")
        print("  bd - Execute backdoor related functionality")

if __name__ == "__main__":
    main()
