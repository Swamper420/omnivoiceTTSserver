#!/usr/bin/env python3
"""
Test Client for OmniVoice TTS Server.
"""

import sys
import json
import urllib.request
import urllib.parse

BASE_URL = "http://localhost:8000"

def test_health():
    print("--- Checking Health Endpoint ---")
    try:
        req = urllib.request.Request(f"{BASE_URL}/health")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print("Health Status:", json.dumps(data, indent=2))
            return data
    except Exception as e:
        print(f"Health check failed: {e}")
        return None

def test_list_voices():
    print("\n--- Listing Voices ---")
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/v1/voices")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            print(f"Found {data.get('count', 0)} voice(s):")
            for v in data.get("voices", []):
                print(f" - ID: {v['voice_id']}, Transcript: '{v['transcript']}', Settings: {v['settings']}")
            return data
    except Exception as e:
        print(f"List voices failed: {e}")
        return None

def test_tts(text="Tervehdys! Tämä on ääni-synteesi testi.", voice="voice_fi", output_filename="output.wav"):
    print(f"\n--- Testing Synthesis for Voice '{voice}' ---")
    payload = {
        "text": text,
        "voice": voice,
        "response_format": "wav",
        "speed": 1.0,
        "num_step": 32,
        "guidance_scale": 2.0
    }
    
    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/tts",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as resp:
            audio_bytes = resp.read()
            with open(output_filename, "wb") as f:
                f.write(audio_bytes)
            print(f"Successfully generated audio ({len(audio_bytes)} bytes) -> saved to '{output_filename}'")
    except Exception as e:
        print(f"TTS synthesis failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python test_client.py [BASE_URL]")
        sys.exit(0)

    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1].rstrip("/")

    test_health()
    test_list_voices()
    print("\nTo test actual TTS generation run the server first: python -m app.main")
