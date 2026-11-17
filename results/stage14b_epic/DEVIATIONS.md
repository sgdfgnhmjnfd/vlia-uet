# Stage 14b Part D (EPIC-KITCHENS-100): skipped

**Part D was not run.** The pre-registered condition was "≥ 12 complete videos from ≥ 6 participants" within 3 hours of the download start (hard stop 2026-11-16 19:03:35). That condition was not met.

- **Download.**
  - Download started 2026-11-16 16:03 with the official `epic_download_scripts` (`epic_video_downloader_verify.py`), 24 pre-registered videos (8 participants × 3).
  - It ran as 1 worker at first, then as 4 parallel workers over disjoint lists.
  - Speed was about 240–360 KiB/s per connection, about 1.1 MB/s in total. Each video is about 2 GB.
- **Complete at the stop.** 4 videos from 2 participants:
  - P01_02, P01_106, P01_11 (P01);
  - P02_118 (P02).
  - That is 8.6 GB.
  - 4 more videos were partial (P02_120, P02_131, P03_101, P03_117).
- **Power loss.** The machine also shut down at 2026-11-16 19:00 (power loss), 3.5 minutes before the hard stop. At ~1.1 MB/s the missing 8 videos (≥ 16 GB) could not have finished in the time left in any case.
- **Not done.** No other data set was tried and no download was restarted after the hard stop. No EPIC unit, feature, sentence or score was computed.
- **Files.** The downloaded files are kept in `/media/dhqg/d3/data/epic/` (CC BY-NC 4.0); none is committed.
- **Code.** The pre-registration (`results/prereg_stage14b_epic.md`) and the code (`scripts/run_stage14b_epic.py`, `scripts/test_stage14b_epic.py`) stay in the repo unchanged, so the part can be run later if the videos are obtained.

## Download continued after the hard stop (2026-11-17, at the user's request)

- **Restart.** On 2026-11-17 at 14:36 the user asked to finish downloading the 24 selected videos. The download was restarted with no time limit (`run_download_resume.sh`, 4 workers, same lists).
- **Partial files lost.** The 4 partial files from 2026-11-16 were resumed with HTTP Range, failed the MD5 check (likely damaged by the power loss), and were deleted by the downloader and restarted from byte 0.
- **Speed.** Real speed is about 0.4–0.5 MiB/s per worker, about 1.9 MiB/s in total.
- **Status.** Part D remains skipped under its pre-registered rule. Running it on videos obtained after the 3-hour limit would be a deviation from the pre-registration, and needs the user's decision.
