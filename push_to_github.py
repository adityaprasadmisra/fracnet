
import os
import subprocess
import sys

# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_DIR = r"E:\fracnet\FracNet"

GITHUB_USERNAME = "AdityaMisra"
GITHUB_EMAIL = "prasadaditya2111@gmail.com"

REPO_URL = "https://github.com/adityaprasadmisra/fracnet.git"

COMMIT_MESSAGE = "Complete FracNet implementation, dataset and results"

# ==========================================
# GIT COMMAND HELPER
# ==========================================

os.chdir(PROJECT_DIR)


def run(command):
    print("\n>>>", " ".join(command))

    result = subprocess.run(command)

    if result.returncode != 0:
        print("\nERROR: Git command failed.")
        sys.exit(result.returncode)


# ==========================================
# CREATE GITIGNORE
# ==========================================

gitignore = """
venv/
.venv/
__pycache__/
*.pyc
.pytest_cache/
.idea/
.vscode/
.DS_Store
Thumbs.db
.env
"""

with open(".gitignore", "w", encoding="utf-8") as f:
    f.write(gitignore.strip() + "\n")

# ==========================================
# VERIFY REQUIRED PROJECT FILES
# ==========================================

required_paths = [
    "models",
    "training",
    "matlab",
    "dataset/fracnet_train.mat",
    "checkpoints/fracnet_best.pth",
    "checkpoints/fracnet_final.pth",
    "results/plots",
    "inference.py",
    "requirements.txt"
]

print("\n========== CHECKING PROJECT ==========")

for path in required_paths:
    if os.path.exists(path):
        print("[OK]", path)
    else:
        print("[MISSING]", path)

        print("\nPlease verify missing files before pushing.")
        sys.exit(1)

# ==========================================
# CHECK LARGE FILES
# ==========================================

print("\n========== CHECKING FILE SIZES ==========")

large_files = []

for root, dirs, files in os.walk(PROJECT_DIR):

    dirs[:] = [
        d for d in dirs
        if d not in ["venv", ".venv", "__pycache__", ".git"]
    ]

    for filename in files:

        filepath = os.path.join(root, filename)

        size_mb = os.path.getsize(filepath) / (1024 * 1024)

        if size_mb > 100:
            large_files.append((filepath, size_mb))

if large_files:
    print("\nFiles larger than GitHub's 100 MB limit:")

    for filepath, size in large_files:
        print(f"{filepath}: {size:.2f} MB")

    print("\nUse Git LFS for these files before pushing.")
    sys.exit(1)

print("No files exceed 100 MB.")

# ==========================================
# INITIALIZE GIT
# ==========================================

if not os.path.exists(".git"):
    run(["git", "init"])

run(["git", "config", "user.name", GITHUB_USERNAME])
run(["git", "config", "user.email", GITHUB_EMAIL])

run(["git", "branch", "-M", "main"])

# ==========================================
# CONFIGURE REMOTE
# ==========================================

remotes = subprocess.run(
    ["git", "remote"],
    capture_output=True,
    text=True
).stdout.split()

if "origin" in remotes:
    run(["git", "remote", "set-url", "origin", REPO_URL])
else:
    run(["git", "remote", "add", "origin", REPO_URL])

# ==========================================
# STAGE PROJECT FILES
# ==========================================

run(["git", "add", "-A"])

print("\n========== FILES TO BE COMMITTED ==========")

run(["git", "status", "--short"])

# ==========================================
# COMMIT
# ==========================================

commit = subprocess.run([
    "git",
    "commit",
    "-m",
    COMMIT_MESSAGE
])

if commit.returncode != 0:
    print("\nCommit failed or no new changes to commit.")
    sys.exit(commit.returncode)

# ==========================================
# PUSH TO GITHUB
# ==========================================

run(["git", "push", "-u", "origin", "main"])

print("\n==========================================")
print("FRACNET PROJECT PUSHED SUCCESSFULLY!")
print("GitHub:", REPO_URL)
print("==========================================")
