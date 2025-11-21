python track.py --input inputs/soccer_footage_1.mp4 `
  --roi 520 110 140 140 `
  --exclude-rect 300 330 600 190 `
  --auto-intensity-percentile 99.9 `
  --intensity-blur 3 `
  --bright-spot --bright-spot-radius 90 --bright-spot-blur 3 --bright-spot-percentile 99.9 --bright-min-area 1 --bright-max-area 80 `
  --max-corners 0 `
  --lk-window 5 --lk-levels 3 `
  --output-video outputs/ball_annotated.mp4 --output-csv outputs/ball_metrics.csv --output-plot outputs/ball_speed.png