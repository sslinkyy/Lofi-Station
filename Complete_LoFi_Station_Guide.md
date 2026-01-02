# 24/7 Lo-Fi Station: Complete Build Guide
## Windows VPS + OBS + Winamp + UTC

---

# TABLE OF CONTENTS

1. [What This Manual Guarantees](#what-this-manual-guarantees)
2. [Legend - How to Read This Manual](#legend)
3. [**DRY RUN CHECKLIST** (Do This First - Before Buying VPS)](#dry-run-checklist)
4. [**FLIGHT CHECKLIST** (Step-by-Step Build)](#flight-checklist)
5. [**DETAILED MANUAL** (Reference Guide)](#detailed-manual)
   - Part 0: Decisions to Lock
   - Part A: VPS Buying Checklist
   - Part B: Windows Hardening
   - Part C: Blender Video Loops
   - Part D: Upload Media to VPS
   - Part E: YouTube Live Setup
   - Part F: OBS Configuration
   - Part G: OBS Scenes
   - Part H: World Clock Overlay
   - Part I: Winamp Setup
   - Part J: Now Playing Integration
   - Part K: Audio Routing
   - Part L: UTC Scene Switching
   - Part M: Music Mood Scheduling
   - Part N: Auto-Start + Crash Recovery
   - Part O: Full Test Plan
   - Part P: Adding New Tracks
6. [Disaster Recovery](#disaster-recovery)
7. [Weekly Maintenance](#weekly-maintenance)

---

## What This Manual Guarantees

If you follow it exactly, you will end up with a Windows VPS that:

- ✅ Streams to YouTube 24/7 with OBS
- ✅ Plays your music with Winamp (shuffle + crossfade)
- ✅ Automatically updates Now Playing in OBS
- ✅ Shows a rotating world clock (UTC + major cities)
- ✅ Automatically switches visual scenes by UTC time
- ✅ Automatically recovers after reboot and most crashes
- ✅ Lets you add new music by dropping files into folders (no rebuilding anything)

---

## Legend

**✅ Checkpoint:** something you must see before continuing
**⚠️ If this fails:** do the fix steps immediately
**📌 Do not change:** settings that must match this guide

---

<a name="dry-run-checklist"></a>
# ✅ PRE-VPS DRY RUN CHECKLIST (DO THIS FIRST)

**Goal:** Prove the system works on your local computer before spending money on a VPS.

**What you'll prove:**
- You can render loop videos correctly
- OBS scenes + overlays work
- Winamp audio captures in OBS
- "Now Playing" updates automatically
- You can simulate UTC scene switching
- 2–4 hour stability test passes

---

## SECTION 1 — Prep Your Test Workspace

☐ Create folders on your computer (exactly):

```
StationDryRun\
  Media\Morning\
  Media\Day\
  Media\Dusk\
  Media\Night\
  Overlays\
  Music\Study\
  Music\Chill\
  Music\Sleep\
  Playlists\
  Scripts\
  Logs\
```

☐ Put 10–20 tracks into each:
  - `Music\Study\`
  - `Music\Chill\`
  - `Music\Sleep\`

**✅ Expected result:** You can see files in those folders in File Explorer.

---

## SECTION 2 — Visual Loop Dry Run (Minimum Viable)

You don't need final 3D yet—just prove looping playback works.

☐ Create four short test loop videos (recommended: 30–60 seconds each) named:
  - `Media\Morning\loop.mp4`
  - `Media\Day\loop.mp4`
  - `Media\Dusk\loop.mp4`
  - `Media\Night\loop.mp4`

(They can be simple—colored backgrounds with text "Morning/Day/Dusk/Night".)

**✅ Expected result:** Each MP4 plays in your media player and loops cleanly.

---

## SECTION 3 — Install and Configure OBS (Local)

☐ Install OBS Studio
☐ Open OBS
☐ Create Profile: `DryRun_720p`
☐ Create Scene Collection: `DryRun_Lofi_UTC`

### Set OBS Settings

☐ Settings → Video:
  - Base: `1280×720`
  - Output: `1280×720`
  - FPS: `30`

☐ Settings → Output (Advanced):
  - Encoder: `x264`
  - Rate Control: `CBR`
  - Bitrate: `3500 Kbps`
  - Keyframe interval: `2`
  - Preset: `veryfast`
  - Audio: `160 Kbps`

**✅ Expected result:** OBS preview works smoothly; no error saving settings.

---

## SECTION 4 — Build the 4 OBS Scenes and Verify Loop Playback

☐ Create scenes:
  - `SCN_Morning`
  - `SCN_Day`
  - `SCN_Dusk`
  - `SCN_Night`

☐ For each scene add a Media Source `BG`:
  - Point to the correct `loop.mp4`
  - Loop = `ON`

**✅ Expected result:** Clicking each scene shows the correct loop video.

---

## SECTION 5 — World Clock Overlay Dry Run

☐ Create `Overlays\world_clock.html` (use the rotating zones HTML below)

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>World Clock</title>
<style>
  html, body { margin:0; padding:0; background:transparent; overflow:hidden;
    font-family: 'Inter','Segoe UI',system-ui,sans-serif; }
  #clock { position:absolute; right:20px; bottom:20px; text-align:right;
    color: rgba(255,255,255,0.85); transition: opacity 1s ease-in-out; }
  #time { font-size: 28px; font-weight: 500; letter-spacing: 1px; }
  #zone { font-size: 14px; opacity: 0.7; margin-top: 2px; }
  .fade-out { opacity: 0; }
</style>
</head>
<body>
  <div id="clock">
    <div id="time">00:00</div>
    <div id="zone">UTC</div>
  </div>
<script>
  const zones = [
    { label: "UTC", timeZone: "UTC" },
    { label: "New York", timeZone: "America/New_York" },
    { label: "London", timeZone: "Europe/London" },
    { label: "Tokyo", timeZone: "Asia/Tokyo" }
  ];
  let currentZone = 0;
  const switchInterval = 20000;
  const timeEl = document.getElementById("time");
  const zoneEl = document.getElementById("zone");
  const clockEl = document.getElementById("clock");
  function updateTime() {
    const now = new Date();
    const zone = zones[currentZone];
    const formatter = new Intl.DateTimeFormat('en-GB', {
      hour:'2-digit', minute:'2-digit', hour12:false, timeZone: zone.timeZone
    });
    timeEl.textContent = formatter.format(now);
    zoneEl.textContent = zone.label;
  }
  function rotateZone() {
    clockEl.classList.add("fade-out");
    setTimeout(() => {
      currentZone = (currentZone + 1) % zones.length;
      updateTime();
      clockEl.classList.remove("fade-out");
    }, 1000);
  }
  updateTime();
  setInterval(updateTime, 1000);
  setInterval(rotateZone, switchInterval);
</script>
</body>
</html>
```

☐ Add it in each scene:
  - Source → Browser → Local file → select `world_clock.html`
  - 400×120, 30 FPS

**✅ Expected result:** Clock appears and rotates zones every ~20 seconds.

---

## SECTION 6 — Install Winamp and Prove Desktop Audio Capture Works

☐ Install Winamp
☐ Open Winamp
☐ In Winamp, play a track

☐ In OBS: Settings → Audio:
  - Desktop Audio = `Default`
  - Mic/Aux = `Disabled`

**✅ Expected result:** OBS "Desktop Audio" meter moves when Winamp plays.

☐ Adjust levels to prevent clipping:
  - Keep peaks around -12 to -6 dB

**✅ Expected result:** No red clipping.

---

## SECTION 7 — Prove "Now Playing" Can Update via Text File

This isolates OBS behavior first.

☐ Create file: `Overlays\nowplaying.txt`
☐ Put: `Starting…`

☐ In OBS (each scene):
  - Add Text (GDI+) → Read from file → select `nowplaying.txt`

☐ Edit `nowplaying.txt` manually (change text) and save.

**✅ Expected result:** OBS text updates after a moment.

⚠️ If this fails, fix OBS text source before moving on.

---

## SECTION 8 — "Now Playing" Automation Dry Run (Plugin Test)

☐ Install your chosen Winamp "Now Playing to file" plugin (WinampNowPlayingToFile or equivalent)
☐ Configure output path to: `Overlays\nowplaying.txt`
☐ Set format to: `Artist – Title`

**✅ Expected result:** When you skip tracks, `nowplaying.txt` changes automatically and OBS overlay updates.

---

## SECTION 9 — Playlist Structure + "Drop-In" Workflow Test

☐ Turn Shuffle ON in Winamp
☐ Enable folder watching / rescanning in Winamp (media library watch folders)

**✅ Expected result:** Winamp detects newly added tracks.

### Drop-In Test

☐ While Winamp is running, copy a new MP3 into `Music\Chill\`

**✅ Expected result:** Within a few minutes (or after manual rescan), that track becomes playable.

---

## SECTION 10 — Simulated UTC Scene Switching

Before you rely on Advanced Scene Switcher, simulate switching:

☐ In OBS, manually switch scenes every 1–2 minutes.

**✅ Expected result:**
- Clock stays visible
- Now playing stays visible
- Background changes
- Audio continues uninterrupted

⚠️ If this fails, fix your scene composition.

---

## SECTION 11 — Stability Test (The Real Dry Run)

☐ Start Winamp (shuffle ON)
☐ Start OBS preview (or record locally)
☐ Let it run for 2–4 hours without touching it.

### During the Run Check:

☐ No audio dropouts
☐ No OBS crash
☐ No runaway CPU usage
☐ Now Playing continues updating
☐ Clock continues rotating zones

**✅ Expected result:** Everything stays stable for hours.

---

## SECTION 12 — "Go Live" Dry Run Without VPS (Optional but Recommended)

☐ Create a YouTube test stream (Unlisted)
☐ Stream from your local computer for 30–60 minutes

**✅ Expected result:** Stream health OK, audio present, overlays visible.

---

## PASS/FAIL Decision

### ✅ PASS if:
- OBS captures Winamp audio reliably
- Now Playing updates automatically
- Loops play without stutter
- Clock overlay works
- 2–4 hour stability test succeeds

### 🚫 FAIL if:
- Desktop Audio doesn't capture consistently
- Now Playing file doesn't update
- OBS crashes or freezes
- Videos stutter badly

**If you fail, fix the issue before buying a VPS. Do not proceed.**

---

<a name="flight-checklist"></a>
# 🛫 24/7 LO-FI STATION — FLIGHT CHECKLIST

(Windows VPS · OBS · Winamp · UTC · Global Audience)

**RULE:** Do not skip steps. Do not reorder steps.
**RULE:** If a checkpoint fails, stop and fix it before continuing.

---

## SECTION 0 — BEFORE YOU START (ONCE)

☐ I have a Windows VPS with RDP access
☐ I can log into the VPS desktop
☐ VPS timezone is set to UTC
☐ I have my music files ready
☐ I have 4 loop videos rendered (Morning / Day / Dusk / Night)

---

## SECTION 1 — VPS BASELINE SETUP (DO FIRST)

### 1.1 Login

☐ Open Remote Desktop Connection
☐ Enter VPS IP
☐ Login successfully

**You should see:** Windows desktop

### 1.2 Set Time to UTC (MANDATORY)

☐ Start → Settings
☐ Time & Language → Date & Time
☐ "Set time automatically" = ON
☐ Time zone = (UTC) Coordinated Universal Time

**You should see:** Clock showing UTC (not your local time)

### 1.3 Disable Sleep

☐ Settings → System → Power & Sleep
☐ Sleep = Never

---

## SECTION 2 — CREATE REQUIRED FOLDERS (NO EXCEPTIONS)

☐ Open File Explorer → C:\
☐ Create the following exactly:

```
C:\Station\
C:\Station\Media\
C:\Station\Media\Morning\
C:\Station\Media\Day\
C:\Station\Media\Dusk\
C:\Station\Media\Night\
C:\Station\Overlays\
C:\Station\Playlists\
C:\Station\Scripts\
C:\Station\Logs\

C:\Music\
C:\Music\Study\
C:\Music\Chill\
C:\Music\Sleep\
```

☐ Double-check spelling and capitalization

---

## SECTION 3 — COPY MEDIA FILES TO VPS

☐ Copy Morning loop → `C:\Station\Media\Morning\loop.mp4`
☐ Copy Day loop → `C:\Station\Media\Day\loop.mp4`
☐ Copy Dusk loop → `C:\Station\Media\Dusk\loop.mp4`
☐ Copy Night loop → `C:\Station\Media\Night\loop.mp4`

☐ Double-click each file to confirm it plays

---

## SECTION 4 — INSTALL OBS (VIDEO ENGINE)

☐ Download OBS Studio
☐ Install OBS
☐ Launch OBS

### 4.1 Create Profile + Scene Collection

☐ OBS → Profile → New → `YT_720p_24_7`
☐ OBS → Scene Collection → New → `Lofi_UTC_24_7`

### 4.2 OBS Output Settings (COPY EXACTLY)

☐ Settings → Output → Mode = Advanced

**Streaming**
☐ Encoder = x264
☐ Rate Control = CBR
☐ Bitrate = 3500 Kbps
☐ Keyframe Interval = 2
☐ CPU Preset = veryfast
☐ Profile = high

**Audio**
☐ Bitrate = 160 Kbps

### 4.3 OBS Video Settings

☐ Base Canvas = 1280×720
☐ Output Resolution = 1280×720
☐ FPS = 30

### 4.4 Stream Key

☐ Settings → Stream
☐ Service = YouTube
☐ Paste Stream Key
☐ Click Apply

---

## SECTION 5 — CREATE OBS SCENES

☐ Create scenes:
  - `SCN_Morning`
  - `SCN_Day`
  - `SCN_Dusk`
  - `SCN_Night`

### 5.1 Add Background Video to Each Scene

**For each scene:**

☐ Select scene
☐ Sources → + → Media Source
☐ Name = BG
☐ Local file → correct loop.mp4
☐ Loop = ON

**You should see:** Correct background video per scene

---

## SECTION 6 — WORLD CLOCK OVERLAY

☐ Create file: `C:\Station\Overlays\world_clock.html`
☐ Paste approved clock HTML (from dry run)
☐ Save as UTF-8

### 6.1 Add Clock to Each Scene

**For each scene:**

☐ Sources → + → Browser
☐ Local File = world_clock.html
☐ Width = 400
☐ Height = 120
☐ FPS = 30
☐ Position bottom-right

**You should see:** Clock + rotating time zones

---

## SECTION 7 — INSTALL WINAMP (AUDIO ENGINE)

☐ Install Winamp
☐ Launch Winamp

### 7.1 Add Music to Drop-In Folders

☐ Study tracks → `C:\Music\Study\`
☐ Chill tracks → `C:\Music\Chill\`
☐ Sleep tracks → `C:\Music\Sleep\`

### 7.2 Enable Folder Watching

☐ Winamp → Ctrl+P
☐ Media Library → Local Media / Watch Folders
☐ Add folder: `C:\Music\`
☐ Enable:
  - Watch folders for changes
  - Rescan on startup

**Test:**
☐ Add a new MP3 → Winamp detects it

### 7.3 Shuffle + Crossfade

☐ Shuffle = ON
☐ Preferences → Playback
☐ Crossfade = ON
☐ Length = 4–6 seconds

---

## SECTION 8 — NOW PLAYING TEXT OUTPUT

☐ Create file: `C:\Station\Overlays\nowplaying.txt`
☐ Put text: `Starting…`

### 8.1 Install Now-Playing Plugin

☐ Close Winamp
☐ Copy plugin files → `Winamp\Plugins\`
☐ Open Winamp
☐ Ctrl+P → Plug-ins → General
☐ Enable plugin
☐ Output path = `nowplaying.txt`
☐ Format = `Artist – Title`

**Test:**
☐ Change track → file updates

### 8.2 Add Now Playing to OBS

**For each scene:**

☐ Sources → + → Text (GDI+)
☐ Read from file
☐ File = `nowplaying.txt`
☐ Font size ~24
☐ Position bottom-left

---

## SECTION 9 — AUDIO ROUTING (OPTION 1: DESKTOP AUDIO)

☐ OBS → Settings → Audio
☐ Desktop Audio = Default
☐ Mic/Aux = Disabled

☐ Play music in Winamp

**You should see:**
☐ Desktop Audio meter moving in OBS

### 9.1 Prevent Clipping

☐ OBS Mixer peaks between -12 dB and -6 dB
☐ No red meters

---

## SECTION 10 — UTC SCENE AUTOMATION

☐ Install Advanced Scene Switcher
☐ Restart OBS

### 10.1 Create Time Rules

☐ 06:00–11:00 → SCN_Morning
☐ 11:00–16:00 → SCN_Day
☐ 16:00–21:00 → SCN_Dusk
☐ 21:00–23:59 → SCN_Night
☐ 00:00–06:00 → SCN_Night

☐ Check interval = 30–60 seconds
☐ Transition = Fade (800–1200 ms)

**Test:**
☐ Change VPS time → scene switches

---

## SECTION 11 — MUSIC MOOD SCHEDULING

☐ Create playlists:
  - `study.m3u`
  - `chill.m3u`
  - `sleep.m3u`

☐ Create BAT files:
  - `load_study.bat`
  - `load_chill.bat`
  - `load_sleep.bat`

☐ Test BAT files manually

☐ Create Task Scheduler tasks:
  - 06:00 UTC → Study
  - 11:00 UTC → Chill
  - 21:00 UTC → Sleep

☐ "Run with highest privileges" = ON
☐ Restart on failure = ON

---

## SECTION 12 — AUTO-START + WATCHDOG

☐ Enable auto-login (netplwiz)
☐ Create OBS startup task with `--startstreaming`
☐ Create OBS watchdog PowerShell script
☐ Test by closing OBS → it restarts

---

## SECTION 13 — FINAL TESTS

### 13.1 Local Test (10 min)

☐ Audio present
☐ Video present
☐ Clock visible
☐ Now Playing updates

### 13.2 YouTube Unlisted Test (24 hours)

☐ Stream health = OK
☐ No dropped frames
☐ Scenes switched
☐ Music switched
☐ No crashes

---

## SECTION 14 — GO LIVE

☐ Set YouTube stream to Public
☐ Lock title / description
☐ Take VPS snapshot

---

## SECTION 15 — DAILY OPERATION (AFTER SETUP)

☐ To add music: drop files into folders
☐ No playlist rebuilds
☐ No OBS changes
☐ No Winamp restarts required

---

## SECTION 16 — EMERGENCY QUICK FIXES

**No Audio?**
☐ Check OBS Desktop Audio meter
☐ Check Winamp playing
☐ Check Windows volume mixer

**No Video?**
☐ Check Media Source path
☐ Uncheck "Close file when inactive"

**No Scene Switching?**
☐ Confirm UTC timezone
☐ Restart OBS

**Everything broken?**
☐ Restore VPS snapshot

---

✅ **END OF FLIGHT CHECKLIST**

---

<a name="detailed-manual"></a>
# DETAILED MANUAL (Reference Guide)

## PART 0 — Decisions to Lock Before You Spend Money

### 0.1 Your Target Quality (Locked)

- **Video:** 1280×720
- **FPS:** 30 stream output (OBS), loops can be 24 fps
- **Streaming bitrate:** 3500 Kbps to start
- **Audio bitrate:** 160 Kbps AAC
- **Scheduling:** UTC for global audience

### 0.2 UTC Schedule (Locked Default)

- **06:00–11:00 UTC** → Morning (Study)
- **11:00–16:00 UTC** → Day (Study/Chill)
- **16:00–21:00 UTC** → Dusk (Chill)
- **21:00–06:00 UTC** → Night (Sleep) (split at midnight in rules)

📌 **Do not improvise times until the system is stable.**

---

## PART A — VPS Buying Checklist

### A.1 You Must Buy a Windows VPS (Not Linux)

Because this build uses:
- OBS + plugins
- Winamp
- Windows Task Scheduler
- RDP remote desktop

### A.2 Minimum VPS Specs (Recommended)

- **OS:** Windows Server 2019 or 2022 (Desktop Experience)
- **CPU:** 2–4 vCPU
- **RAM:** 8 GB
- **Disk:** 80–120 GB SSD
- **Bandwidth:** ideally ≥ 3 TB/month, preferably 5–10 TB/month or "unmetered"

#### A.2.1 Bandwidth Sanity Check (Important)

If you stream at 3500 Kbps:
- **Per second:** 3500 kilobits
- **Per hour:** 3500 × 3600 = 12,600,000 kilobits ≈ 1.575 GB/hour
- **Per day:** 1.575 × 24 ≈ 37.8 GB/day
- **Per month:** 37.8 × 30 ≈ 1.13 TB/month

So 3 TB/month gives you headroom.

### A.3 Optional: GPU

You do not need a GPU for pre-rendered loops.
Only add GPU later if you plan real-time visualizers.

**✅ Checkpoint:**
- ☐ Windows VPS chosen
- ☐ Bandwidth limit understood
- ☐ You have RDP credentials (IP / user / password)

---

## PART B — First Login: Windows Hardening for 24/7

### B.1 Connect via Remote Desktop (RDP)

On your normal PC:
1. Press Windows key → type **Remote Desktop Connection**
2. Open it
3. In "Computer" enter the VPS IP
4. Click **Connect**
5. Enter username/password
6. Accept certificate prompt if shown

**✅ Checkpoint:**
- ☐ You see the VPS desktop

### B.2 Set Timezone to UTC (MANDATORY)

On the VPS:
1. Click **Start**
2. Click **Settings** (gear icon)
3. Click **Time & Language**
4. Click **Date & time**
5. Turn ON: **Set time automatically**
6. Set **Time zone** to: **(UTC) Coordinated Universal Time**

**✅ Checkpoint:**
- ☐ Clock shows UTC time (not your local time)

### B.3 Disable Sleep (If Any)

1. Settings → System → Power & sleep
2. Set **Sleep** to **Never** (if present)

**✅ Checkpoint:**
- ☐ No sleep timer

### B.4 Create Your Station Folders

1. Open **File Explorer** → This PC → Local Disk (C:)
2. Create exactly:

```
C:\Station\
C:\Station\Media\
C:\Station\Media\Morning\
C:\Station\Media\Day\
C:\Station\Media\Dusk\
C:\Station\Media\Night\
C:\Station\Overlays\
C:\Station\Playlists\
C:\Station\Scripts\
C:\Station\Logs\

C:\Music\
C:\Music\Study\
C:\Music\Chill\
C:\Music\Sleep\
```

**✅ Checkpoint:**
- ☐ All folders exist with exact spelling

---

## PART C — Blender: Create the 4 Video Loops

**This runs on your normal computer, not the VPS.**

### C.1 Install Blender

1. Download Blender from the official Blender site
2. Install
3. Open Blender

**✅ Checkpoint:**
- ☐ Blender opens successfully

### C.2 Create a Project and Save Immediately

1. File → New → General
2. File → Save As…
3. Name: `LofiRoom_Master.blend`

**✅ Checkpoint:**
- ☐ You have a .blend saved

### C.3 Set Your Exact Loop Duration (5 Minutes)

We want 5:00 minutes at 24 fps:
- 5 minutes = 300 seconds
- 300 × 24 = **7200 frames**

In Blender:
1. At the bottom **Timeline**:
   - **Start:** 1
   - **End:** 7200
2. **Output Properties** (printer icon):
   - **Frame Rate:** 24

**✅ Checkpoint:**
- ☐ End frame is 7200
- ☐ FPS is 24

### C.4 Set Output Resolution to 720p

**Output Properties:**
- Resolution X = 1280
- Resolution Y = 720

**✅ Checkpoint:**
- ☐ 1280×720 set

### C.5 Choose Render Engine (Use Eevee First)

**Render Properties** (camera icon):
- **Render Engine:** Eevee

**Why:** Fastest iteration, stylized looks good.

**✅ Checkpoint:**
- ☐ Eevee selected

### C.6 Create a "Simple Room" First

You only need:
- Walls + floor
- Bed block
- Window block
- Lamp block

You can replace with Meshy assets later.

**✅ Checkpoint:**
- ☐ You can render a still image of a room

### C.7 Camera Setup

1. Press **Numpad 0** (camera view)
2. Select camera
3. Move/rotate until your framing matches your keyframe vibe
4. **Camera Properties:**
   - **Lens:** 35–40mm

**✅ Checkpoint:**
- ☐ Camera shows bed + window + subject area

### C.8 Make the Loop "Alive" but Safe

Choose simple animations that loop:
- Rain outside window (scrolling texture)
- Lamp flicker (tiny)
- Ambient light shift (tiny)
- Optional micro-camera motion (tiny)

📌 **RULE:** Frame 1 and frame 7200 must match.

If you animate with Noise modifiers, ensure they tile cleanly or use looping curves.

**✅ Checkpoint:**
- ☐ Scrub frame 1 → 7200 and nothing "pops"

### C.9 Export Settings (MP4 H.264, No Audio)

**Output Properties:**
- **File Format:** FFmpeg video
- **Encoding:**
  - **Container:** MPEG-4
  - **Video Codec:** H.264
  - **Audio:** OFF (no audio track)
- **Bitrate:**
  - Target 2000–3000 kbps (this is just the loop file)

### C.10 Create 4 Lighting Variants (Morning/Day/Dusk/Night)

Easiest foolproof method:

Save 4 copies of the project:
- `LofiRoom_Morning.blend`
- `LofiRoom_Day.blend`
- `LofiRoom_Dusk.blend`
- `LofiRoom_Night.blend`

In each copy, change ONLY lighting + outside brightness.

**Suggested differences:**
- **Morning:** Brighter window, slightly softer lamp
- **Day:** Neutral, least dramatic
- **Dusk:** Warm lamp + cool window (main vibe)
- **Night:** Darker outside, stronger city lights, lamp slightly dimmer

**✅ Checkpoint:**
- ☐ You can render one frame from each and clearly see the difference

### C.11 Render the 4 Loops

Render → Render Animation in each .blend, output as:
- `morning_loop.mp4`
- `day_loop.mp4`
- `dusk_loop.mp4`
- `night_loop.mp4`

**✅ Checkpoint:**
- ☐ Each file is ~5 minutes and plays correctly

---

## PART D — Upload Media to VPS

### D.1 Copy Files via RDP Clipboard (Simplest)

1. On your PC, locate the 4 MP4 files
2. Copy them
3. In the VPS File Explorer, paste into:
   - `C:\Station\Media\Morning\`
   - `C:\Station\Media\Day\`
   - `C:\Station\Media\Dusk\`
   - `C:\Station\Media\Night\`
4. Then rename inside each folder to `loop.mp4`:
   - `C:\Station\Media\Morning\loop.mp4`
   - etc.

**✅ Checkpoint:**
- ☐ Each folder contains `loop.mp4` and it plays in Windows Media Player

---

## PART E — YouTube Live Setup

On your normal computer:
1. Open **YouTube Studio**
2. Click **Create** → **Go live**
3. Choose **Stream**
4. Copy your **Stream Key** (keep it private)

**✅ Checkpoint:**
- ☐ You have the stream key ready to paste into OBS

---

## PART F — OBS: Install + Configure Like a Broadcast Engineer

### F.1 Install OBS on the VPS

1. Download OBS Studio
2. Install
3. Launch OBS

**✅ Checkpoint:**
- ☐ OBS opens

### F.2 Create a Dedicated Profile and Scene Collection

Top menu:
- **Profile** → New → `YT_720p_24_7`
- **Scene Collection** → New → `Lofi_UTC_24_7`

**✅ Checkpoint:**
- ☐ You see the new profile + collection selected

### F.3 OBS Output Settings (YouTube Stable)

**OBS → Settings → Output:**

**Output Mode:** Advanced

**Streaming:**
- Encoder: x264
- Rate Control: CBR
- Bitrate: 3500 Kbps
- Keyframe Interval: 2
- CPU Usage Preset: veryfast
- Profile: high

**Audio:**
- Audio Bitrate: 160

**OBS → Settings → Video:**
- Base (Canvas): 1280×720
- Output (Scaled): 1280×720
- FPS: 30

**✅ Checkpoint:**
- ☐ Bitrate/CBR/Keyframe=2 set
- ☐ 1280×720 output set

### F.4 Set Stream Key

**OBS → Settings → Stream:**
- Service: YouTube (or Custom)
- Stream Key: paste it

**✅ Checkpoint:**
- ☐ No error when you click Apply

---

## PART G — OBS Scenes: 4 Time-of-Day Loops

### G.1 Create 4 Scenes (Exact Names)

In **Scenes** box:
- `SCN_Morning`
- `SCN_Day`
- `SCN_Dusk`
- `SCN_Night`

**✅ Checkpoint:**
- ☐ All 4 scenes exist

### G.2 Add Media Source "BG" in Each Scene

For each scene:
1. Click the scene name
2. Sources → + → Media Source
3. Name: `BG`
4. Check **Loop**
5. Browse to the correct `loop.mp4`

**✅ Checkpoint:**
- ☐ Switching scenes changes background video

---

## PART H — Rotating World Clock Overlay (HTML Browser Source)

### H.1 Create the World Clock File

On VPS:
1. Open **Notepad**
2. Paste the HTML (from Section 5 of Dry Run)
3. Save As:
   - **File:** `C:\Station\Overlays\world_clock.html`
   - **Save as type:** All Files
   - **Encoding:** UTF-8

**✅ Checkpoint:**
- ☐ File exists at `C:\Station\Overlays\world_clock.html`

### H.2 Add Clock to Each OBS Scene

For each scene:
1. Sources → + → Browser
2. Check **Local file**
3. Select `C:\Station\Overlays\world_clock.html`
4. Width: 400, Height: 120, FPS: 30
5. Drag bottom-right

**✅ Checkpoint:**
- ☐ Clock shows and rotates zones every ~20s

**⚠️ If it shows a blank box:**
- Right-click Browser source → Properties → reselect the HTML
- Ensure file extension is `.html` not `.txt`
- Ensure it's saved UTF-8

---

## PART I — Winamp: Drop-In Folders, Shuffle, Crossfade

### I.1 Install Winamp on VPS

1. Install and open it.

**✅ Checkpoint:**
- ☐ Winamp opens

### I.2 Copy Music Into Folders

Copy your tracks into:
- `C:\Music\Study\`
- `C:\Music\Chill\`
- `C:\Music\Sleep\`

**✅ Checkpoint:**
- ☐ You see files in those folders in File Explorer

### I.3 Enable Winamp Media Library Scanning

In Winamp:
1. Press **Ctrl + P** (Preferences)
2. Find **Media Library** settings
3. Add watched folder: `C:\Music\`
4. Enable:
   - Watch folders for changes
   - Rescan on startup

**✅ Checkpoint:**
- ☐ Add one test MP3 into `C:\Music\Chill\` and Winamp detects it within a few minutes (or after a manual rescan)

### I.4 Enable Shuffle

In Winamp main window:
- Click **Shuffle** so it's ON

**✅ Checkpoint:**
- ☐ Shuffle button appears active

### I.5 Enable Crossfade

In Preferences:
- Playback → enable **Crossfade**
- Set to 4–6 seconds

**✅ Checkpoint:**
- ☐ Tracks blend smoothly

---

## PART J — Now Playing: Winamp → Text File → OBS

### J.1 Create the Output File Location

Create (or ensure exists):
- `C:\Station\Overlays\nowplaying.txt`

Put a placeholder line:
```
Starting…
```

**✅ Checkpoint:**
- ☐ You can open this file and see the text

### J.2 Install a "Now Playing to File" Winamp Plugin

Use a plugin that writes metadata to a text file.

#### J.2.1 Install Approach (Generic but Foolproof)

1. Close Winamp completely
2. Copy the plugin DLL files into Winamp's Plugins folder (commonly):
   - `C:\Program Files (x86)\Winamp\Plugins\`
3. Open Winamp
4. **Ctrl+P** → Plug-ins → General Purpose
5. Find the plugin
6. Set output path:
   - `C:\Station\Overlays\nowplaying.txt`
7. Set format:
   - `Artist – Title`

**✅ Checkpoint:**
- ☐ When a track changes, `nowplaying.txt` changes

**⚠️ If the file never updates:**
- Confirm Winamp is playing
- Confirm plugin enabled
- Confirm file path correct and you have permission to write to `C:\Station\Overlays\`

### J.3 Add Now Playing to Each OBS Scene

For each scene:
1. Sources → + → Text (GDI+)
2. Check "Read from file"
3. Select `C:\Station\Overlays\nowplaying.txt`
4. Font size 24, white, slight shadow
5. Place bottom-left

**✅ Checkpoint:**
- ☐ When Winamp changes tracks, OBS text changes

---

## PART K — Audio in OBS (Option 1: Desktop Audio Capture)

### K.1 Configure OBS Audio Devices

**OBS → Settings → Audio:**
- Desktop Audio: Default
- Mic/Aux: Disabled
- Click Apply.

**✅ Checkpoint:**
- ☐ In OBS Mixer, "Desktop Audio" meter moves when Winamp plays

**⚠️ If meter does not move:**
- In Winamp, ensure music is playing
- In Windows volume mixer, ensure Winamp is not muted
- OBS Settings → Audio → set Desktop Audio to the actual audio device (not Default)

### K.2 Prevent Clipping

In OBS Mixer:
- Keep peaks between -12 dB and -6 dB
- If it hits red: reduce Winamp volume or lower Desktop Audio slider

**✅ Checkpoint:**
- ☐ No red clipping

---

## PART L — UTC Scene Switching (Advanced Scene Switcher)

### L.1 Install Plugin

Install **Advanced Scene Switcher** and restart OBS.

### L.2 Create Time Rules (UTC)

**Rules:**
- 06:00–11:00 → SCN_Morning
- 11:00–16:00 → SCN_Day
- 16:00–21:00 → SCN_Dusk
- 21:00–23:59 → SCN_Night
- 00:00–06:00 → SCN_Night

**Set:**
- Check interval: 30–60 seconds
- Transition: Fade 800–1200 ms

**✅ Checkpoint:**
- ☐ Change VPS clock temporarily to 16:01 UTC and OBS switches to Dusk within a minute

---

## PART M — Music Mood Switching by UTC

### M.1 Create Playlist Files

Create:
- `C:\Station\Playlists\study.m3u`
- `C:\Station\Playlists\chill.m3u`
- `C:\Station\Playlists\sleep.m3u`

**If you don't know how to create M3U:**
1. Open Winamp
2. Load files from `C:\Music\Study\`
3. Save playlist → save to `C:\Station\Playlists\study.m3u`
4. Repeat for Chill and Sleep.

**✅ Checkpoint:**
- ☐ Those 3 playlist files exist

### M.2 Scheduling Playlist Swaps (Task Scheduler)

We will schedule at:
- 06:00 UTC → Study
- 11:00 UTC → Chill
- 21:00 UTC → Sleep

#### M.2.1 Create Helper Batch Files

Create:
- `C:\Station\Scripts\load_study.bat`
- `C:\Station\Scripts\load_chill.bat`
- `C:\Station\Scripts\load_sleep.bat`

Open Notepad and paste into each:

**load_study.bat**
```batch
@echo off
start "" "C:\Program Files (x86)\Winamp\winamp.exe" /LOADPLAYLIST="C:\Station\Playlists\study.m3u" /PLAY
```

**load_chill.bat**
```batch
@echo off
start "" "C:\Program Files (x86)\Winamp\winamp.exe" /LOADPLAYLIST="C:\Station\Playlists\chill.m3u" /PLAY
```

**load_sleep.bat**
```batch
@echo off
start "" "C:\Program Files (x86)\Winamp\winamp.exe" /LOADPLAYLIST="C:\Station\Playlists\sleep.m3u" /PLAY
```

**✅ Checkpoint:**
- ☐ Double-click each BAT and Winamp loads/plays that playlist

**⚠️ If double-clicking does nothing:**
- Your Winamp install path differs. Find `winamp.exe`:
  - Right-click Winamp shortcut → Open file location
  - Update the BAT paths accordingly.

#### M.2.2 Create Task Scheduler Tasks (3 Tasks)

1. Open Start → type **Task Scheduler**
2. Click **Create Task** (not "Basic Task")

**For each task:**

**General tab**
- Name: `Winamp - Load Study`
- Check: **Run with highest privileges**
- Check: **Run whether user is logged on or not**

**Triggers tab**
- New…
- Begin the task: **On a schedule**
- **Daily**
- Start time: **06:00:00**
- Ensure timezone is UTC

**Actions tab**
- New…
- Action: **Start a program**
- Program/script: `C:\Station\Scripts\load_study.bat`

**Settings tab**
- If the task fails, restart every: **1 minute**
- Attempt to restart up to: **3 times**

Repeat for Chill (11:00) and Sleep (21:00).

**✅ Checkpoint:**
- ☐ In Task Scheduler, right-click a task → Run → Winamp switches playlists

---

## PART N — Auto-Start + Crash Recovery

### N.1 Enable Windows Auto-Login

1. Press **Win+R**
2. Type: `netplwiz`
3. Press Enter
4. Uncheck: "Users must enter a user name and password…"
5. Apply
6. Enter password when prompted
7. Reboot VPS

**✅ Checkpoint:**
- ☐ After reboot, you land on desktop without typing password

### N.2 Create "Start OBS Streaming" Task

#### N.2.1 Find OBS Executable Path

Usually:
```
C:\Program Files\obs-studio\bin\64bit\obs64.exe
```

#### N.2.2 Create Task Scheduler Task: Start OBS at Logon

**Task Scheduler → Create Task:**

**General**
- Name: `OBS - Start Streaming`
- Run with highest privileges
- Run whether user is logged on or not

**Trigger**
- At log on (your user)

**Action**
- Start a program:
  - Program/script: `C:\Program Files\obs-studio\bin\64bit\obs64.exe`
  - Add arguments: `--startstreaming`

**✅ Checkpoint:**
- ☐ Reboot VPS → OBS launches and starts streaming automatically

### N.3 Add a Watchdog That Relaunches OBS If It Crashes

Create file:
- `C:\Station\Scripts\obs_watchdog.ps1`

Paste:
```powershell
$log = "C:\Station\Logs\watchdog.log"
"--- watchdog start $(Get-Date) ---" | Out-File -Append $log

while ($true) {
  $obs = Get-Process obs64 -ErrorAction SilentlyContinue
  if (-not $obs) {
    "$(Get-Date) OBS not running -> starting" | Out-File -Append $log
    Start-Process "C:\Program Files\obs-studio\bin\64bit\obs64.exe" "--startstreaming"
  }
  Start-Sleep -Seconds 30
}
```

**Create Task Scheduler task: OBS - Watchdog**

**Trigger**
- At startup

**Action**
- Program: `powershell.exe`
- Arguments: `-ExecutionPolicy Bypass -File C:\Station\Scripts\obs_watchdog.ps1`

**✅ Checkpoint:**
- ☐ Close OBS manually → it restarts within ~30 seconds

---

## PART O — Full Test Plan

### O.1 10-Minute Component Test (Before Going Live)

1. Start Winamp → confirm music plays
2. OBS → confirm:
   - Background video visible
   - Clock visible
   - Now playing visible
   - Desktop Audio meter moves

**✅ Checkpoint:**
- ☐ You can see and hear everything in OBS preview

### O.2 30-Minute Live Test (Unlisted)

In YouTube Studio:
1. Set stream visibility to **Unlisted**
2. Start OBS streaming
3. Watch in YouTube preview:
   - Audio present
   - No stutters
   - Overlays show

**✅ Checkpoint:**
- ☐ YouTube shows "excellent" or acceptable stream health

### O.3 UTC Rule Test (Fast Method)

Temporarily change VPS clock to:
- 15:59 → should be Day
- 16:01 → should switch to Dusk within 60s
- Then set it back to correct time.

**✅ Checkpoint:**
- ☐ Scene switching works

### O.4 24-Hour Stability Test (Required)

Run Unlisted for 24 hours.

Check:
- OBS Stats: dropped frames near zero
- Winamp still playing
- Overlays still updating
- Scene switching occurred

**✅ Checkpoint:**
- ☐ 24 hours clean = ready to go Public

---

## PART P — Adding New Tracks

**You never "rebuild the system."**

### P.1 Add Tracks

1. Upload new files to VPS
2. Drop them into one folder:
   - `C:\Music\Study\`
   - `C:\Music\Chill\`
   - `C:\Music\Sleep\`
3. Wait a few minutes for Winamp to detect (or trigger a rescan in Winamp)

**✅ Checkpoint:**
- ☐ You can skip tracks and eventually hit the new one

---

<a name="disaster-recovery"></a>
# DISASTER RECOVERY — What to Do When It Breaks at 3am

## C1 — Stream Went Offline

1. RDP into VPS
2. Is OBS running?
   - If NO → run OBS manually
3. Is Winamp running?
   - If NO → start Winamp
4. If both running but no stream:
   - Stop stream in OBS
   - Start stream again
5. If still broken:
   - Reboot VPS

## C2 — No Audio on YouTube

1. In OBS, look at "Desktop Audio" meter
2. If not moving → Winamp not playing or wrong audio device
3. If moving but YouTube silent:
   - Wait 15 seconds (YouTube delay)
   - Check OBS is not muted
   - Check Windows volume mixer

## C3 — Now Playing Stuck

1. Open `C:\Station\Overlays\nowplaying.txt`
2. If file isn't changing:
   - Restart Winamp
   - Confirm plugin enabled
3. If file changes but OBS doesn't:
   - In OBS, open Text source properties and reselect the file path

## C4 — Clock Disappeared

1. Right-click browser source → Properties
2. Re-check Local file path
3. Reload source

## C5 — Scenes Not Switching

1. Confirm Windows timezone still UTC
2. Open Advanced Scene Switcher rules
3. Restart OBS

## C6 — Nuclear Option

Restore VPS snapshot/backups.

---

<a name="weekly-maintenance"></a>
# WEEKLY MAINTENANCE SOP

Once per week:

☐ Check YouTube stream health
☐ Check OBS Stats (View → Stats)
☐ Check disk space (C: drive)
☐ Add new tracks

**Optional planned reboot:**
1. Reboot
2. Confirm OBS auto-streams
3. Confirm Winamp runs
4. Confirm overlays update

---

# FINAL NOTES

This guide has been designed to be **foolproof** and **resilient**. If you follow the steps exactly:

1. **Start with the Dry Run Checklist** to validate everything works locally
2. **Use the Flight Checklist** for step-by-step VPS setup
3. **Reference the Detailed Manual** when you need explanations or troubleshooting

**When you're done:**
- Your stream runs 24/7 automatically
- Adding music is drag-and-drop
- Recovery from crashes is automatic
- Scene and mood switching happens by UTC time

**Good luck, and happy streaming!** 🎵
