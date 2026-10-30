"""Direct GitHub API Pusher (Zero dependencies - pure standard library).

Uploads all repository files directly to GitHub via the GitHub REST API
using the provided Personal Access Token (PAT).
Works completely without git.exe installed on the machine!
"""

import os
import sys
import json
import base64
import urllib.request
import urllib.error

TOKEN = "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"
REPO = "yashsidana/Retail_demand"
API_BASE = f"https://api.github.com/repos/{REPO}"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def api_request(url: str, method: str = "GET", data: dict = None) -> dict:
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "Antigravity-Agent",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    body = json.dumps(data).encode("utf-8") if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            resp_body = resp.read().decode("utf-8")
            return json.loads(resp_body) if resp_body else {}
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8")
        print(f"HTTP Error {e.code} on {method} {url}: {error_msg}")
        raise


def get_default_branch_and_head():
    """Gets default branch and latest commit SHA."""
    try:
        repo_info = api_request(API_BASE, "GET")
        default_branch = repo_info.get("default_branch", "main")
    except Exception:
        default_branch = "main"

    try:
        ref_info = api_request(f"{API_BASE}/git/ref/heads/{default_branch}", "GET")
        commit_sha = ref_info["object"]["sha"]
        return default_branch, commit_sha
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # Empty repository without commits
            return default_branch, None
        raise


def get_all_project_files():
    """Collects all project files to upload, ignoring caches and binaries."""
    ignore_dirs = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "env", ".ipynb_checkpoints"}
    ignore_exts = {".pyc", ".pyo", ".pyd", ".parquet", ".db"}
    
    files_to_upload = []
    for root, dirs, files in os.walk(ROOT_DIR):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in ignore_exts or f in [".DS_Store", "Thumbs.db"]:
                continue
            
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, ROOT_DIR).replace("\\", "/")
            files_to_upload.append((rel_path, full_path))

    return files_to_upload


def upload_via_git_data_api(branch: str, parent_sha: str, files: list):
    """Creates Blobs -> Tree -> Commit -> Updates Ref in one single atomic commit."""
    print(f"Creating Git Blobs for {len(files)} files...")
    tree_items = []
    
    for rel_path, full_path in files:
        with open(full_path, "rb") as f:
            content = f.read()
            b64_content = base64.b64encode(content).decode("utf-8")

        blob_resp = api_request(f"{API_BASE}/git/blobs", "POST", {
            "content": b64_content,
            "encoding": "base64"
        })
        tree_items.append({
            "path": rel_path,
            "mode": "100644",
            "type": "blob",
            "sha": blob_resp["sha"]
        })
        print(f"  + Blob created: {rel_path}")

    print("Creating new Git Tree...")
    tree_data = {"tree": tree_items}
    if parent_sha:
        # Base tree
        try:
            parent_commit = api_request(f"{API_BASE}/git/commits/{parent_sha}", "GET")
            tree_data["base_tree"] = parent_commit["tree"]["sha"]
        except Exception:
            pass

    tree_resp = api_request(f"{API_BASE}/git/trees", "POST", tree_data)
    tree_sha = tree_resp["sha"]

    print("Creating Commit...")
    commit_data = {
        "message": "feat: complete retail demand forecasting & inventory optimization project",
        "tree": tree_sha,
        "parents": [parent_sha] if parent_sha else []
    }
    commit_resp = api_request(f"{API_BASE}/git/commits", "POST", commit_data)
    new_commit_sha = commit_resp["sha"]

    print(f"Updating branch '{branch}' ref to new commit {new_commit_sha[:7]}...")
    if parent_sha:
        api_request(f"{API_BASE}/git/refs/heads/{branch}", "PATCH", {
            "sha": new_commit_sha,
            "force": True
        })
    else:
        api_request(f"{API_BASE}/git/refs", "POST", {
            "ref": f"refs/heads/{branch}",
            "sha": new_commit_sha
        })

    print(f"\nSUCCESS! All files committed and pushed to https://github.com/{REPO} (branch: {branch})")


def upload_via_contents_api(branch: str, files: list):
    """Fallback: upload files one by one using Contents API."""
    print("Using Contents API to push files...")
    for rel_path, full_path in files:
        with open(full_path, "rb") as f:
            b64_content = base64.b64encode(f.read()).decode("utf-8")

        # Check if file already exists
        sha = None
        try:
            existing = api_request(f"{API_BASE}/contents/{rel_path}?ref={branch}", "GET")
            sha = existing.get("sha")
        except Exception:
            pass

        data = {
            "message": f"Add {rel_path}",
            "content": b64_content,
            "branch": branch
        }
        if sha:
            data["sha"] = sha

        api_request(f"{API_BASE}/contents/{rel_path}", "PUT", data)
        print(f"  + Uploaded: {rel_path}")

    print(f"\nSUCCESS! All files uploaded to https://github.com/{REPO}")


def main():
    print(f"Connecting to GitHub API for {REPO}...")
    try:
        branch, parent_sha = get_default_branch_and_head()
    except Exception as e:
        print(f"Could not connect to repo: {e}")
        branch, parent_sha = "main", None

    files = get_all_project_files()
    print(f"Found {len(files)} files to push.\n")

    try:
        upload_via_git_data_api(branch, parent_sha, files)
    except Exception as e:
        print(f"Git Data API failed ({e}), falling back to Contents API...")
        upload_via_contents_api(branch, files)


if __name__ == "__main__":
    main()

