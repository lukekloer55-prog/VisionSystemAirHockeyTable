#!/usr/bin/env python3
"""Full-table yellow puck tracker with goal/contact shot-planning overlays.

Run on the Jetson desktop: python3 airhockey_shot_tracker.py
Run mathematical tests without a camera: python3 airhockey_shot_tracker.py --self-test
All measurements and dynamics use millimetres and seconds.
"""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import time


def predict_path(position, velocity, duration, width, length, radius):
    """Propagate the puck CENTER, reflecting at side rails, stopping at ends.

    Left/right rails are x=0 and x=width. The center collides at radius
    and width-radius, not at the rail itself. End lines do NOT reflect.
    Returns position, velocity, path vertices, bounce data, and end time.
    """
    x, y = map(float, position)
    vx, vy = map(float, velocity)
    left, right = radius, width - radius
    top, bottom = radius, length - radius
    # Bound tiny measurement excursions to the legal center rectangle.
    x = max(left, min(right, x))
    y = max(top, min(bottom, y))
    path = [(x, y)]
    bounces = []
    remaining = max(0.0, duration)
    elapsed = 0.0
    while remaining > 1e-10:
        # Calculate time to the next side wall in the direction of travel.
        tx = ((right - x) / vx if vx > 0 else
              (left - x) / vx if vx < 0 else math.inf)
        # An end crossing stops prediction: goal/end-wall rules are unknown.
        ty = ((bottom - y) / vy if vy > 0 else
              (top - y) / vy if vy < 0 else math.inf)
        tx, ty = max(0.0, tx), max(0.0, ty)
        step = min(remaining, tx, ty)
        x, y = x + vx * step, y + vy * step
        elapsed += step
        remaining -= step
        path.append((x, y))
        if ty <= step + 1e-10:
            return (x, y), (vx, vy), path, bounces, elapsed
        if tx <= step + 1e-10:
            # Reflection formula v' = v - 2*(v dot n)*n.
            # Here the side-wall normal n is horizontal: reverse only vx.
            incoming_angle = math.degrees(math.atan2(vy, vx))
            vx = -vx
            outgoing_angle = math.degrees(math.atan2(vy, vx))
            bounces.append((elapsed, (x, y), incoming_angle, outgoing_angle))
    return (x, y), (vx, vy), path, bounces, None


def self_test():
    """Check reflection timing, radius offsets, multiple hits, and end exits."""
    def close(actual, expected):
        assert all(abs(a - b) < 1e-7 for a, b in zip(actual, expected)), (actual, expected)
    # Width=100 and radius=5: center limits are x=5 and x=95.
    result = predict_path((50, 50), (10, 20), 0.1, 100, 200, 5)
    close(result[0], (51, 52))
    result = predict_path((90, 50), (100, 20), 0.1, 100, 200, 5)
    close(result[0], (90, 52))
    close(result[1], (-100, 20))
    assert len(result[3]) == 1
    result = predict_path((10, 50), (-100, 0), 0.1, 100, 200, 5)
    close(result[0], (10, 50))
    result = predict_path((50, 50), (1000, 0), 0.3, 100, 200, 5)
    close(result[0], (20, 50))
    assert len(result[3]) == 3
    result = predict_path((95, 50), (100, 0), 0.1, 100, 200, 5)
    close(result[0], (85, 50))
    result = predict_path((50, 190), (0, 100), 0.1, 100, 200, 5)
    close(result[0], (50, 195))
    assert abs(result[4] - 0.05) < 1e-7
    result = predict_path((50, 50), (0, 0), 0.1, 100, 200, 5)
    close(result[0], (50, 50))
    # Many cases checked against an independent triangular-wave solution.
    for start in (5, 25, 70, 95):
        for speed in (-2500, -100, 0, 100, 2500):
            for dt in (0.001, 0.03, 0.25, 1.0):
                phase = (start - 5 + speed * dt) % 180
                expected = 5 + (phase if phase <= 90 else 180 - phase)
                result = predict_path((start, 50), (speed, 0), dt, 100, 200, 5)
                close(result[0], (expected, 50))
    print('PASS: 87 trajectory cases, including multiple side bounces and end exits.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--camera-mode', choices=('full30', 'full21', 'fast60'), default='full30',
                        help='IMX219: full sensor aspect at 30/21 FPS, or cropped 720p at 60 FPS')
    # Separate file: never reinterpret the earlier half-table calibration as full-table.
    parser.add_argument('--config', default=str(Path(__file__).with_name('full_table_calibration.json')))
    parser.add_argument('--goal-config', default=str(Path(__file__).with_name('shot_goal.json')))
    parser.add_argument('--shot-speed-mm-s', type=float, default=1500)
    parser.add_argument('--paddle-restitution', type=float, default=0.8)
    parser.add_argument('--paddle-diameter-mm', type=float, default=None)
    parser.add_argument('--goal-width-mm', type=float, default=254)
    parser.add_argument('--goal-margin-mm', type=float, default=5)
    parser.add_argument('--recalibrate', action='store_true')
    parser.add_argument('--horizon-ms', type=float, default=30)
    parser.add_argument('--preview-ms', type=float, default=500)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        from shot_planner import self_test as shot_tests
        shot_tests()
        return
    if not 0 < args.horizon_ms <= 1000 or not 0 < args.preview_ms <= 2000:
        parser.error('Use horizon 1..1000 ms and preview 1..2000 ms.')
    if not math.isfinite(args.shot_speed_mm_s) or args.shot_speed_mm_s <= 0:
        parser.error('Shot speed must be positive and finite.')
    if not math.isfinite(args.paddle_restitution) or not 0 <= args.paddle_restitution <= 1:
        parser.error('Paddle restitution must be between 0 and 1.')
    if args.paddle_diameter_mm is not None and (not math.isfinite(args.paddle_diameter_mm) or args.paddle_diameter_mm <= 0):
        parser.error('Paddle diameter must be positive and finite.')
    if not math.isfinite(args.goal_margin_mm) or args.goal_margin_mm < 0:
        parser.error('Goal margin must be nonnegative and finite.')

    # Import imaging libraries only for the camera app, not the math tests.
    import cv2
    import numpy as np

    # Read saved calibration if present; never silently use an invalid file.
    config_path = Path(args.config).expanduser()
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except (ValueError, OSError) as error:
            raise SystemExit(f'Cannot read calibration: {error}. Use a new --config path.')

    def ask_number(prompt, default):
        # Positive dimensions are required so angles use the correct aspect ratio.
        while True:
            try:
                response = input(f'{prompt} [{default:g}; ENTER keeps this]: ').strip()
                value = float(response) if response else default
                if math.isfinite(value) and value > 0:
                    return value
            except ValueError:
                pass
            print('Enter a positive number in millimetres.')

    dimensions = config.get('dimensions')
    if dimensions is None or args.recalibrate:
        print('Calibrate the ENTIRE table. Far end / opponent goal is y=0.')
        # Defaults are your measurements converted from inches to millimetres.
        # Full length 77 in: two equal 38.5-inch halves.
        width = ask_number('Width between left/right side rails (mm)', 1049.3375)
        length = ask_number('FULL length between end rails (mm)', 1955.8)
        diameter = ask_number('Puck diameter (mm)', 71.7042)
    else:
        width, length, diameter = dimensions
    if not all(math.isfinite(v) and v > 0 for v in (width, length, diameter)) or diameter >= min(width, length):
        raise SystemExit('Invalid dimensions: puck diameter must be smaller than width and length.')
    radius = diameter / 2
    if not math.isfinite(args.goal_width_mm) or not diameter + 2*args.goal_margin_mm < args.goal_width_mm <= width:
        parser.error('Goal must fit inside the width and clear puck diameter plus margins.')

    # Use one common scale on both axes so reflected angles are not distorted.
    # Longest dimension of the display is roughly 800 pixels.
    scale = 800.0 / max(width, length)
    out_width, out_height = int(math.ceil(width * scale)) + 1, int(math.ceil(length * scale)) + 1
    # Modes come from this Jetson's reported IMX219 Argus sensor-mode list.
    # 1640/1232 == 3280/2464 == 205/154: preserve the EXACT sensor aspect.
    # Use even output dimensions for NV12 conversion and reduce detector workload.
    modes = {
        'full30': (3, 1640, 1232, 30, [820, 616]),
        'full21': (0, 3280, 2464, 21, [820, 616]),
        'fast60': (4, 1280, 720, 60, [960, 540]),
    }
    sensor_mode, capture_width, capture_height, capture_fps, raw_size = modes[args.camera_mode]
    corners = config.get('corners', [None] * 4)
    # Matching image dimensions alone cannot prove the same field of view.
    if (args.recalibrate or config.get('raw_size') != raw_size
            or config.get('camera') != args.camera
            or config.get('camera_mode') != args.camera_mode):
        corners = [None] * 4
    names = ['1 FAR LEFT', '2 FAR RIGHT', '3 NEAR RIGHT', '4 NEAR LEFT']
    transform = None
    calibrating = any(c is None for c in corners)

    def make_transform(points):
        # Puck centers touch two rails at each corner. Map them to radius-inset
        # positions, not the physical rail intersections. This avoids late bounces.
        source = np.asarray(points, np.float32)
        if source.shape != (4, 2) or not np.isfinite(source).all():
            raise ValueError('Invalid corner coordinates.')
        if not cv2.isContourConvex(source.reshape(-1, 1, 2)) or abs(cv2.contourArea(source)) < 2000:
            raise ValueError('Corners must surround a sizeable rectangle in order 1,2,3,4.')
        destination = np.float32([
            [radius, radius], [width-radius, radius],
            [width-radius, length-radius], [radius, length-radius]
        ]) * scale
        matrix = cv2.getPerspectiveTransform(source, destination)
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10:
            raise ValueError('Degenerate calibration; recapture the four corners.')
        return matrix

    if not calibrating:
        try:
            transform = make_transform(corners)
        except ValueError as error:
            print(error)
            calibrating = True
            corners = [None] * 4

    # HSV sliders let you tune yellow under the actual table lighting.
    # Allow border dragging/maximization while preserving displayed image geometry.
    # The warped table keeps physical table proportions, not camera proportions.
    window_flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    cv2.namedWindow('Yellow mask', window_flags)
    cv2.resizeWindow('Yellow mask', 600, 450)
    slider_names = ['H low', 'H high', 'S low', 'S high', 'V low', 'V high']
    saved_hsv = config.get('hsv', [20, 40, 90, 255, 70, 255])
    for name, value, maximum in zip(slider_names, saved_hsv, [179, 179, 255, 255, 255, 255]):
        cv2.createTrackbar(name, 'Yellow mask', int(value), maximum, lambda value: None)

    def hsv_values():
        return [cv2.getTrackbarPos(name, 'Yellow mask') for name in slider_names]

    def yellow_mask(image):
        # White pixels are yellow candidates. No closing by default: avoid merging.
        lo_h, hi_h, lo_s, hi_s, lo_v, hi_v = hsv_values()
        return cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
                           np.array([lo_h, lo_s, lo_v], np.uint8),
                           np.array([hi_h, hi_s, hi_v], np.uint8))

    def detect(mask, calibrated):
        # In the top-down view, expected puck area follows from its real diameter.
        expected_area = math.pi * (radius * scale) ** 2
        results = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area < 12 or perimeter <= 0:
                continue
            circularity = 4 * math.pi * area / perimeter**2
            # Raw calibration views may show very oval pucks due to perspective.
            if circularity < (0.45 if calibrated else 0.15):
                continue
            if calibrated and not 0.30 * expected_area <= area <= 2.0 * expected_area:
                continue
            moments = cv2.moments(contour)
            center = np.array([moments['m10']/moments['m00'], moments['m01']/moments['m00']])
            results.append((center, area, circularity, contour))
        return results

    def label(image, text, y, color=(255, 255, 255)):
        # Black outline keeps labels visible over light table surfaces.
        for thickness, ink in [(3, (0, 0, 0)), (1, color)]:
            cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.44, ink, thickness)

    def pixel(point):
        return tuple(int(round(v * scale)) for v in point)

    def save():
        # JSON is plain text. It preserves settings across program restarts.
        data = {'dimensions': [width, length, diameter], 'corners': corners,
                'raw_size': raw_size, 'camera': args.camera,
                'camera_mode': args.camera_mode, 'hsv': hsv_values()}
        config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = config_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2))
        temporary.replace(config_path)
        print(f'Saved {config_path}')

    # A click chooses which yellow blob is the puck during calibration.
    selection = {'point': None}
    stable_centers = deque(maxlen=12)
    def on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            selection['point'] = np.array([x, y], dtype=float)
            stable_centers.clear()
    cv2.namedWindow('Camera - calibration', window_flags)
    cv2.resizeWindow('Camera - calibration', raw_size[0], raw_size[1])
    cv2.setMouseCallback('Camera - calibration', on_click)

    # Add shot overlay without changing the existing detector or motion forecast.
    from shot_overlay import ShotOverlay
    shot_overlay = ShotOverlay(cv2, width, length, radius, scale, args.goal_config,
                               args.shot_speed_mm_s, args.paddle_restitution,
                               args.paddle_diameter_mm, args.goal_margin_mm,
                               args.goal_width_mm)

    # Choose an actual capture mode; stretching a 720p image cannot recover crops.
    pipeline = (
        f'nvarguscamerasrc sensor-id={args.camera} sensor-mode={sensor_mode} ! '
        f'video/x-raw(memory:NVMM),width={capture_width},height={capture_height},'
        f'format=NV12,framerate={capture_fps}/1 ! '
        f'nvvidconv ! video/x-raw,width={raw_size[0]},height={raw_size[1]},format=BGRx ! '
        'videoconvert ! video/x-raw,format=BGR ! '
        'appsink max-buffers=1 drop=true sync=false'
    )
    camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not camera.isOpened():
        camera.release()
        cv2.destroyAllWindows()
        raise SystemExit('Could not open CSI camera. Close other camera apps first.')

    last_position = last_seen = None
    velocity = np.zeros(2)
    velocity_ready = False
    lost_timeout = 0.05
    fps = 0.0
    previous_time = time.monotonic()
    notice = 'Place puck at a corner, click its center, then press its number.'
    print('Q quit | C recalibrate | 1..4 corners | ENTER apply | S save | G goal | H shot mode')
    print('Orange path and magenta contact X are IDEAL aim proposals, not motor commands.')
    print(f'Camera {args.camera_mode}: {capture_width}x{capture_height} @ {capture_fps} FPS; '
          f'processing {raw_size[0]}x{raw_size[1]}. Drag window edges to resize.')
    print('Far end is y=0. Left/right rails reflect; near/far ends STOP the prediction.')
    try:
        while True:
            ok, raw = camera.read()
            now = time.monotonic()
            if not ok:
                print('Camera stopped delivering frames.')
                break
            dt_frame = now - previous_time
            previous_time = now
            if dt_frame > 0:
                fps = 0.9 * fps + 0.1 / dt_frame
            raw_display = raw.copy()
            if calibrating:
                mask = yellow_mask(raw)
                candidates = detect(mask, False)
                chosen = None
                if selection['point'] is not None and candidates:
                    chosen = min(candidates, key=lambda c: np.linalg.norm(c[0] - selection['point']))
                    if np.linalg.norm(chosen[0] - selection['point']) > 60:
                        chosen = None
                if chosen is not None:
                    stable_centers.append(chosen[0])
                    cv2.drawContours(raw_display, [chosen[3]], -1, (0, 255, 0), 2)
                    cv2.circle(raw_display, tuple(chosen[0].astype(int)), 4, (0, 0, 255), -1)
                else:
                    stable_centers.clear()
                for index, corner in enumerate(corners):
                    if corner is not None:
                        p = tuple(np.rint(corner).astype(int))
                        cv2.circle(raw_display, p, 6, (255, 255, 0), 2)
                        cv2.putText(raw_display, str(index+1), p, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                label(raw_display, 'CALIBRATION: 1 far-left | 2 far-right | 3 near-right | 4 near-left', 22)
                label(raw_display, 'Puck touching BOTH rails; click puck, hold still, press corner number.', 44)
                label(raw_display, notice, 66, (0, 255, 255))
                label(raw_display, 'ENTER applies all four | S saves yellow settings | Q quits', 88)
                cv2.imshow('Yellow mask', mask)
            else:
                # Rectify BEFORE detection and velocity estimation. Both axes now
                # have the same mm/pixel scale, so reflection geometry is meaningful.
                topdown = cv2.warpPerspective(raw, transform, (out_width, out_height))
                mask = yellow_mask(topdown)
                candidates = detect(mask, True)
                measured = None
                expected = None
                age = now - last_seen if last_seen is not None else math.inf
                if last_position is not None and age <= lost_timeout:
                    expected = predict_path(last_position, velocity, age, width, length, radius)[0]
                valid = []
                for candidate in candidates:
                    center = candidate[0] / scale
                    # Reject detections substantially outside legal center positions.
                    if not radius*0.5 <= center[0] <= width-radius*0.5 or not radius*0.5 <= center[1] <= length-radius*0.5:
                        continue
                    distance = np.linalg.norm(center - expected) if expected is not None else 0
                    gate = max(diameter * 2, 40) + np.linalg.norm(velocity) * 0.025
                    if expected is not None and distance > gate:
                        continue
                    expected_area = math.pi * (radius*scale)**2
                    score = distance/gate + abs(math.log(candidate[1]/expected_area))
                    valid.append((score, center, candidate))
                if valid:
                    _, measured, chosen = min(valid, key=lambda item: item[0])
                    cv2.drawContours(topdown, [chosen[3]], -1, (0,255,0), 2)

                position = None
                status = 'LOST - no prediction'
                if measured is not None:
                    if last_seen is not None and 0 < age <= lost_timeout:
                        new_velocity = (measured-last_position)/age
                        if velocity_ready:
                            # Avoid retaining the old horizontal sign after a rebound.
                            if velocity[0]*new_velocity[0] < 0 and abs(new_velocity[0]) > 50:
                                velocity[0] = new_velocity[0]
                            velocity = 0.5 * new_velocity + 0.5 * velocity
                        else:
                            velocity = new_velocity
                            velocity_ready = True
                    else:
                        velocity = np.zeros(2)
                        velocity_ready = False
                    # These assignments run for EVERY valid detection.
                    last_position, last_seen = measured.copy(), now
                    position = measured
                    status = 'MEASURED'
                    cv2.circle(topdown, pixel(position), 4, (0,255,0), -1)
                elif last_seen is not None and age <= lost_timeout and velocity_ready:
                    estimate = predict_path(last_position, velocity, age, width, length, radius)
                    if estimate[4] is None:
                        position = np.array(estimate[0])
                        status = 'ESTIMATED ONLY'
                        cv2.circle(topdown, pixel(position), 8, (0,255,255), 2)
                else:
                    last_position = last_seen = None
                    velocity_ready = False
                    velocity = np.zeros(2)

                label(topdown, f'{status} | FPS {fps:.1f}', 20, (0,255,0) if measured is not None else (0,255,255))
                if position is not None:
                    label(topdown, f'Now X={position[0]:.1f} Y={position[1]:.1f} mm', 40)
                if position is not None and velocity_ready:
                    # If we already bounced during a missing-frame interval, use
                    # that interval's reflected velocity for the future forecast.
                    prediction_velocity = estimate[1] if measured is None else velocity
                    short = predict_path(position, prediction_velocity, args.horizon_ms/1000, width, length, radius)
                    preview = predict_path(position, prediction_velocity, args.preview_ms/1000, width, length, radius)
                    # Cyan is the longer optional path; red is the short-horizon marker.
                    for start, end in zip(preview[2], preview[2][1:]):
                        cv2.line(topdown, pixel(start), pixel(end), (255,255,0), 2)
                    for bounce in preview[3]:
                        cv2.circle(topdown, pixel(bounce[1]), 5, (255,0,255), -1)
                    if short[4] is None:
                        cv2.drawMarker(topdown, pixel(short[0]), (0,0,255), cv2.MARKER_CROSS, 18, 2)
                        label(topdown, f'+{args.horizon_ms:g}ms X={short[0][0]:.1f} Y={short[0][1]:.1f} mm', 60, (0,0,255))
                    else:
                        label(topdown, f'End reached in {short[4]*1000:.0f}ms; forecast stops', 60, (0,0,255))
                    label(topdown, f'Vx={prediction_velocity[0]:.0f} Vy={prediction_velocity[1]:.0f} mm/s', 80)
                    if preview[3]:
                        bounce = preview[3][0]
                        label(topdown, f'Bounce {bounce[0]*1000:.0f}ms: {bounce[2]:.0f} -> {bounce[3]:.0f} deg', 100, (255,0,255))
                    label(topdown, f'Cyan path: up to {args.preview_ms:g}ms (ideal motion)', 120)
                # Plan only from a fresh measurement and an established velocity.
                shot_overlay.set_calibration(corners)
                topdown = shot_overlay.draw(topdown, position, velocity if velocity_ready else None,
                                           measured is not None)
                label(raw_display, 'Tracking: C calibrate | S save | G goal | H shot mode | Q quit', 22)
                cv2.imshow('Top-down tracker', topdown)
                cv2.imshow('Yellow mask', mask)

            # Display and key handling happen exactly ONCE per camera frame.
            cv2.imshow('Camera - calibration', raw_display)
            key = cv2.waitKey(1) & 0xff
            if not calibrating:
                shot_overlay.key(key)
            if key == ord('q'):
                break
            if key == ord('c'):
                calibrating = True
                corners = [None]*4
                stable_centers.clear()
                selection['point'] = None
                last_position = last_seen = None
                velocity_ready = False
                notice = 'New calibration: place puck, click it, then press corner number.'
            if key == ord('s'):
                save()
            if calibrating and key in map(ord, '1234'):
                if len(stable_centers) < stable_centers.maxlen:
                    notice = 'Click the puck and hold it still for at least 12 frames first.'
                elif np.max(np.std(np.array(stable_centers), axis=0)) > 2.5:
                    notice = 'Puck moved: hold it still, then press the number again.'
                else:
                    index = key - ord('1')
                    corners[index] = np.mean(np.array(stable_centers), axis=0).tolist()
                    notice = f'Saved {names[index]}. Move to next corner and click puck again.'
                    selection['point'] = None
                    stable_centers.clear()
            if calibrating and key in (10, 13):
                if any(c is None for c in corners):
                    notice = 'Capture all four corner positions before pressing ENTER.'
                else:
                    try:
                        transform = make_transform(corners)
                        save()
                        calibrating = False
                        last_position = last_seen = None
                        velocity_ready = False
                    except ValueError as error:
                        notice = str(error)
    except KeyboardInterrupt:
        print('Stopping tracker.')
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
