#!/bin/bash
# Double-click this file in Finder to start the digit recognizer.
cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/Current/bin:$PATH"

fail() {
  echo
  echo "$1"
  echo
  read -n 1 -s -r -p "Press any key to close this window..."
  exit 1
}

echo "=== Digit Recognizer ==="

# 1. Find Python 3.10 or newer.
PY=""
for cand in python3 python3.13 python3.12 python3.11 python3.10; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(sys.version_info < (3, 10))' >/dev/null 2>&1; then
    PY="$cand"; break
  fi
done
[ -n "$PY" ] || fail "Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/ and double-click this file again."

# 2. Private Python environment for this app (made once; remade if it was copied from another computer).
if ! .venv/bin/python -c 'import sys' >/dev/null 2>&1; then
  echo "Setting up for first use (one time only)..."
  rm -rf .venv
  "$PY" -m venv .venv || fail "Could not create the Python environment."
fi

# 3. Install packages (again only if requirements.txt changed).
if [ ! -f .venv/installed.txt ] || ! cmp -s requirements.txt .venv/installed.txt; then
  echo "Installing packages..."
  .venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt \
    || fail "Package install failed. Check your internet connection and try again."
  cp requirements.txt .venv/installed.txt
fi

# 4. Train the model the first time (downloads MNIST, about 2 minutes).
if [ ! -f model.joblib ]; then
  echo "Training the recognizer for the first time (about 2 minutes)..."
  .venv/bin/python train.py || fail "Training failed."
fi

# 5. Start the app and open the browser.
.venv/bin/python app.py --open || fail "The app stopped with an error."
