# Deploying ZEKO VRE to Raspberry Pi 5 via SSH

This guide details how to securely transfer and run the **ZEKO Voice Response Engine** on a Raspberry Pi 5 (8 GB RAM, ARM64) over an SSH connection.

## ⚠️ Hardware Prerequisites
> **Audio Note:** The Raspberry Pi 5 does **not** have an onboard 3.5mm audio jack or built-in microphone. You must connect a **USB microphone and speaker** (or a USB Audio Dongle / I2S HAT) before running the engine.

---

## Step 1: Transfer the Project Files
From your Windows/Host machine, use `scp` to copy the cleaned project directory to your Pi. (The project has been minimized to ~80 KB for instant transfer).

```powershell
scp -r D:\Projects\ZEKO-Communications pi@<raspberry-pi-ip>:~/ZEKO-Communications
```
*(Replace `<raspberry-pi-ip>` with your Pi's actual IP address, e.g., `192.168.1.50`)*

---

## Step 2: SSH into the Pi & Install System Libraries
Open a terminal and connect to your Pi:

```bash
ssh pi@<raspberry-pi-ip>
```

Install the required Linux build tools, audio headers, and `tmux` (for background sessions):

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip portaudio19-dev libasound2-dev ffmpeg libopenblas-dev cmake build-essential git tmux

# Give your user permission to access audio hardware:
sudo usermod -aG audio $USER
```

---

## Step 3: Configure the Python Environment
Navigate to the transferred project and create a virtual environment:

```bash
cd ~/ZEKO-Communications
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Install Python Dependencies
For the best performance on the Pi's ARM CPU, we compile `llama-cpp-python` with OpenBLAS acceleration:

```bash
# 1. Install standard dependencies (includes lightweight ONNX runtime for VAD)
pip install -r requirements.txt

# 2. Re-install llama-cpp-python with hardware acceleration
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

---

## Step 4: Download the Offline LLM Model
The engine is configured to use the highly efficient Gemma 3 1B model for offline fallback. Download it directly into the `models/` directory:

```bash
mkdir -p models
curl -L -o models/gemma-3-1b-it-Q4_K_M.gguf \
  https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/resolve/main/google_gemma-3-1b-it-Q4_K_M.gguf
```

---

## Step 5: Configure API Keys
Copy the example environment configuration and add your Gemini API key:

```bash
cp .env.example .env
nano .env
```
Add your key to `GEMINI_API_KEY=your_key_here`.
*(Press `Ctrl+O` -> `Enter` -> `Ctrl+X` to save and exit)*

---

## Step 6: Test Audio Devices over SSH
By default, running audio apps over SSH can fail if the session doesn't link to the system's PipeWire/PulseAudio bus. 

Run the following command to verify your USB mic/speaker are detected:
```bash
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
python3 -m sounddevice
```

---

## Step 7: Run the Engine (Backgrounded with Tmux)
To ensure the robot doesn't immediately shut down when you close your laptop or the SSH connection drops, use `tmux`:

1. **Start a persistent session:**
   ```bash
   tmux new -s zeko
   ```
2. **Launch the engine:**
   ```bash
   chmod +x run.sh
   ./run.sh
   ```
3. **Detach safely:**
   Press `Ctrl + B`, release, then press `D`. (You can now safely close your SSH terminal).
4. **Re-attach later:**
   To check the logs or stop the engine, SSH back into the Pi and run:
   ```bash
   tmux attach -t zeko
   ```
