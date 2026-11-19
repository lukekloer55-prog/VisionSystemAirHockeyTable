"""OpenCV goal selection and ideal shot overlays for the calibrated tracker."""
import json
import math
from pathlib import Path

from shot_planner import plan_shot


class ShotOverlay:
    def __init__(self, cv2, width, length, radius, scale, config_path, speed,
                 restitution, paddle_diameter, margin, goal_width):
        self.cv = cv2
        self.width, self.length, self.radius, self.scale = width, length, radius, scale
        self.file = Path(config_path)
        self.speed, self.restitution, self.margin = speed, restitution, margin
        self.paddle_radius = paddle_diameter / 2 if paddle_diameter else None
        # User's goal is centred and 254 mm wide; no click measurement required.
        self.default_goal = [(width-goal_width)/2, (width+goal_width)/2, 0.0]
        self.goal = None
        self.fingerprint = None
        self.saved = {}
        self.mode = 'direct'
        self.selecting = False
        self.clicks = []
        self.message = f'Centred {goal_width:g} mm far goal | G override | H shot'
        try:
            self.saved = json.loads(self.file.read_text()) if self.file.exists() else {}
        except (OSError, ValueError):
            self.message = 'Cannot read goal file; press G to mark again'
        cv2.namedWindow('Top-down tracker', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Top-down tracker', 680, 810)
        cv2.setMouseCallback('Top-down tracker', self.on_click)

    def set_calibration(self, corners):
        # A moved camera or changed calibration invalidates clicked goal positions.
        fp = dict(dimensions=[self.width, self.length, self.radius], corners=corners)
        if fp != self.fingerprint:
            self.fingerprint = fp
            self.goal = self.default_goal.copy()
            self.clicks = []
            self.selecting = False
            if self.saved.get('calibration') == fp:
                candidate = self.saved.get('goal')
                try:
                    # Validate the saved aperture using the same planner as live shots.
                    plan_shot((self.width/2, self.length/2), (0, 0), self.width,
                              self.radius, candidate, margin=self.margin)
                    if len(candidate) == 3 and candidate[2] == 0:
                        self.goal = candidate
                except (TypeError, ValueError):
                    pass

    def key(self, key):
        if key == ord('g'):
            self.selecting, self.clicks = True, []
            self.message = 'Click inside LEFT then RIGHT goal post at FAR end (y=0)'
        elif key == 27:
            self.selecting, self.clicks = False, []
            self.message = 'Goal selection cancelled'
        elif key == ord('h'):
            modes = ['direct', 'left', 'right']
            self.mode = modes[(modes.index(self.mode)+1) % len(modes)]

    def on_click(self, event, x, y, flags, userdata):
        if event != self.cv.EVENT_LBUTTONDOWN or not self.selecting:
            return
        point = (x/self.scale, y/self.scale)
        # The first version supports a goal on the far end, not an arbitrary line.
        if not 0 <= point[0] <= self.width or not 0 <= point[1] <= max(40, self.radius):
            self.message = 'Click goal posts near the FAR edge; y is snapped to 0'
            return
        self.clicks.append(point)
        if len(self.clicks) == 1:
            self.message = 'Now click inside RIGHT goal post'
            return
        left, right = self.clicks[0][0], self.clicks[1][0]
        candidate = [left, right, 0.0]
        try:
            plan_shot((self.width/2, self.length/2), (0, 0), self.width,
                      self.radius, candidate, margin=self.margin)
            self.file.parent.mkdir(parents=True, exist_ok=True)
            payload = dict(goal=candidate, calibration=self.fingerprint)
            tmp = self.file.with_suffix('.tmp')
            tmp.write_text(json.dumps(payload, indent=2))
            tmp.replace(self.file)
            self.saved, self.goal = payload, candidate
            self.message = f'Goal saved: X={left:.0f}..{right:.0f} mm, Y=0'
            self.selecting = False
        except (ValueError, OSError) as error:
            self.message = str(error)
        self.clicks = []

    def draw(self, image, position, incoming, measured):
        cv = self.cv
        # Add a readable panel without painting over the goal or tracked puck.
        # Top-left image coordinates stay unchanged, so mouse goal selection works.
        image = cv.copyMakeBorder(image, 0, 155, 0, max(0, 800-image.shape[1]),
                                  cv.BORDER_CONSTANT, value=(25,25,25))
        # Use local helpers to keep all physical-to-pixel conversions consistent.
        def px(point):
            return tuple(int(round(v*self.scale)) for v in point)

        def text(value, row, color=(255,255,255)):
            # A bottom panel separates shot controls from existing velocity labels.
            y = image.shape[0] - 135 + row*19
            for thickness, ink in ((3,(0,0,0)), (1,color)):
                cv.putText(image, value, (8,y), cv.FONT_HERSHEY_SIMPLEX, 0.38, ink, thickness)

        def segment(a, b, color, thickness=2):
            cv.line(image, px(a), px(b), color, thickness)

        text('IDEAL AIM ONLY | G goal | H direct/left/right | Esc cancel', 0)
        text(self.message, 1)
        if self.goal:
            left, right, gy = self.goal
            segment((left,gy+2/self.scale), (right,gy+2/self.scale), (0,255,0), 4)
        for point in self.clicks:
            cv.drawMarker(image, px(point), (0,255,255), cv.MARKER_CROSS, 14, 2)
        if self.selecting:
            return image
        if not self.goal:
            text('Mark the opponent goal in the FULL-TABLE calibrated view.', 2)
            return image
        # Do not create a confident contact target from extrapolated/lost frames.
        if not measured or position is None or incoming is None:
            text('Shot unavailable: waiting for measured puck + velocity.', 2)
            return image
        try:
            shot = plan_shot(position, incoming, self.width, self.radius, self.goal,
                             self.mode, self.speed, self.restitution,
                             self.paddle_radius, self.margin)
        except ValueError as error:
            text(str(error), 2)
            return image
        # Orange path is the requested post-impact centre trajectory.
        for a,b in zip(shot['path'],shot['path'][1:]):
            segment(a,b,(0,165,255))
        if len(shot['path']) == 3:
            cv.circle(image,px(shot['path'][1]),5,(255,0,255),-1)
        cv.drawMarker(image,px(shot['path'][-1]),(0,255,0),cv.MARKER_CROSS,16,2)
        contact, normal, tangent = shot['contact'], shot['normal'], shot['tangent']
        # White tangent is perpendicular to the contact radius (normal).
        span = self.radius * 1.2
        segment(tuple(c-span*t for c,t in zip(contact,tangent)),
                tuple(c+span*t for c,t in zip(contact,tangent)),(255,255,255))
        segment(contact,position,(255,0,255))
        cv.circle(image,px(position),max(1,int(self.radius*self.scale)),(255,0,255),1)
        cv.drawMarker(image,px(contact),(255,0,255),cv.MARKER_TILTED_CROSS,18,3)
        # Incoming arrow shows measured travel, not a proposed paddle command.
        vnorm = math.hypot(*incoming)
        if vnorm > 1:
            start = tuple(p-v/vnorm*self.radius*2 for p,v in zip(position,incoming))
            cv.arrowedLine(image,px(start),px(position),(255,200,0),2,tipLength=0.2)
        # Draw the required paddle normal velocity as a signed directional arrow.
        u = shot['paddle_normal_speed']
        start = tuple(c-normal[i]*self.radius*2 for i,c in enumerate(contact))
        end = tuple(s+normal[i]*math.copysign(self.radius,u) for i,s in enumerate(start))
        if abs(u)>1:
            cv.arrowedLine(image,px(start),px(end),(0,255,255),2,tipLength=0.3)
        warning = 'Paddle size unset; arm reach NOT checked'
        if shot['paddle_center']:
            pc, pr = shot['paddle_center'], self.paddle_radius
            cv.circle(image,px(pc),max(1,int(pr*self.scale)),(0,255,255),1)
            if not pr <= pc[0] <= self.width-pr or not pr <= pc[1] <= self.length-pr:
                warning = 'PADDLE OVERLAPS RAIL/END: contact is not feasible'
            else:
                warning = f'Paddle center {pc[0]:.0f},{pc[1]:.0f} mm; reach NOT checked'
        text(f'{self.mode.upper()} X-contact {contact[0]:.0f},{contact[1]:.0f} mm',2,(255,0,255))
        text(f'Tangent {shot["tangent_deg"]:.1f} deg | launch {shot["outgoing_deg"]:.1f} deg',3)
        text(f'Paddle normal {u:.0f} mm/s | shot {self.speed:.0f} mm/s',4,(0,255,255))
        text(warning,5,(0,255,255))
        def angle(value):
            return 'n/a' if value is None else f'{value:.1f}'
        text(f'Paddle-frame normal angles: in {angle(shot["incidence_deg"])} / out {angle(shot["departure_deg"])} deg',6)
        return image
