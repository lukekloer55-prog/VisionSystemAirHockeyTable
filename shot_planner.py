"""Ideal direct/one-bank shot geometry, in millimetres and seconds.

No camera dependencies. The paddle is a prescribed-velocity, frictionless
surface; this is an aiming model, not a verified robot motion command.
"""
import math


def unit(vector):
    """Return a direction of length one, rejecting a zero-length vector."""
    length = math.hypot(*vector)
    if length < 1e-9:
        raise ValueError('No direction for a zero vector')
    return tuple(v / length for v in vector)


def plan_shot(position, incoming, width, radius, goal, mode='direct',
              speed=1500.0, restitution=0.8, paddle_radius=None, margin=5.0):
    """Aim at the safe middle of a horizontal far-end goal.

    goal = (left inside post X, right inside post X, goal-line Y).
    A bank reflects the target about the radius-offset CENTER collision line.
    Contact normal follows desired velocity minus incoming velocity, rather
    than simply pointing at the goal. Tangential velocity is unchanged.
    """
    values = (*position, *incoming, width, radius, *goal, speed, restitution, margin)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Shot inputs must be finite')
    if speed <= 0 or radius <= 0 or width <= 2 * radius or margin < 0:
        raise ValueError('Invalid speed, radius, width or margin')
    if not 0 <= restitution <= 1:
        raise ValueError('Restitution must be between zero and one')
    if paddle_radius is not None and (not math.isfinite(paddle_radius) or paddle_radius <= 0):
        raise ValueError('Paddle radius must be positive')
    if mode not in ('direct', 'left', 'right'):
        raise ValueError('Unknown shot mode')
    x, y = position
    left, right, gy = goal
    if not 0 <= left < right <= width:
        raise ValueError('Goal posts must lie inside the table width')
    # Shrink the aperture by puck radius plus a configurable safety margin.
    safe_left, safe_right = left + radius + margin, right - radius - margin
    if safe_left >= safe_right:
        raise ValueError('Goal opening is too narrow for the puck and margin')
    if not radius <= x <= width - radius or y <= gy + radius:
        raise ValueError('Puck is outside the shot-planning region')
    target = ((safe_left + safe_right) / 2, gy)
    path = [tuple(position)]
    if mode != 'direct':
        wall = radius if mode == 'left' else width - radius
        # Mirror the goal across the puck-centre rail boundary.
        mirrored = (2 * wall - target[0], gy)
        dx = mirrored[0] - x
        fraction = (wall - x) / dx if abs(dx) > 1e-9 else -1
        if not 0 < fraction < 1:
            raise ValueError('No valid single-wall path from this position')
        bounce = (wall, y + fraction * (gy - y))
        if bounce[1] <= gy + radius:
            raise ValueError('Bank is too close to the end rail')
        path.append(bounce)
    path.append(target)
    # Check clearance when the puck first touches the end-rail plane, not just
    # when its centre crosses the goal line. This conservatively avoids posts.
    start_final = path[-2]
    fraction = (gy + radius - start_final[1]) / (gy - start_final[1])
    entry_x = start_final[0] + fraction * (target[0] - start_final[0])
    if not safe_left <= entry_x <= safe_right:
        raise ValueError('Shot approaches too shallowly to clear the goal posts')
    direction = unit((path[1][0] - x, path[1][1] - y))
    outgoing = tuple(speed * d for d in direction)
    delta = tuple(w - v for w, v in zip(outgoing, incoming))
    impulse_per_mass = math.hypot(*delta)
    if impulse_per_mass < 1e-6:
        raise ValueError('Puck already has the requested velocity; no strike needed')
    normal = unit(delta)  # Points from the paddle towards the puck centre.
    tangent = (-normal[1], normal[0])
    contact = (x - radius * normal[0], y - radius * normal[1])
    vin_normal = sum(v * n for v, n in zip(incoming, normal))
    # v_out = v_in - (1+e) ((v_in-u) dot n) n.
    # Solve for the paddle's required normal speed u dot n.
    paddle_normal_speed = vin_normal + impulse_per_mass / (1 + restitution)
    # Angles measured from the normal in the moving paddle's reference frame.
    # Incidence equals departure only for an ideal elastic (e=1) contact.
    def normal_angle(vel):
        relative = tuple(v-paddle_normal_speed*n for v,n in zip(vel,normal))
        magnitude = math.hypot(*relative)
        if magnitude < 1e-9:
            return None
        cosine = abs(sum(v*n for v,n in zip(relative,normal))) / magnitude
        return math.degrees(math.acos(min(1, max(0, cosine))))
    paddle_center = None
    if paddle_radius is not None:
        paddle_center = tuple(p - (radius + paddle_radius) * n
                              for p, n in zip(position, normal))
    path_length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    return dict(path=path, contact=contact, normal=normal, tangent=tangent,
                incoming=tuple(incoming), outgoing=outgoing,
                paddle_center=paddle_center, paddle_radius=paddle_radius,
                paddle_normal_speed=paddle_normal_speed,
                tangent_deg=math.degrees(math.atan2(tangent[1], tangent[0])) % 180,
                outgoing_deg=math.degrees(math.atan2(direction[1], direction[0])),
                incidence_deg=normal_angle(incoming), departure_deg=normal_angle(outgoing),
                flight_ms=1000 * path_length / speed,
                safe_goal=(safe_left, safe_right, gy), mode=mode)


def self_test():
    """Verify finite-radius banks and reconstruct the moving-paddle collision."""
    goal = (350, 650, 0)
    count = 0
    for pos in ((500, 1200), (100, 500), (900, 1500)):
        for velocity in ((0, 0), (300, 800), (-700, -200)):
            for mode in ('direct', 'left', 'right'):
                for e in (0, 0.8, 1):
                    s = plan_shot(pos, velocity, 1000, 35, goal, mode, 1500, e, 40)
                    assert abs(math.dist(pos, s['contact']) - 35) < 1e-7
                    assert abs(math.dist(pos, s['paddle_center']) - 75) < 1e-7
                    n, t, u = s['normal'], s['tangent'], s['paddle_normal_speed']
                    assert abs(sum(a*b for a,b in zip(n,t))) < 1e-9
                    relative_normal = sum(v*a for v,a in zip(velocity,n)) - u
                    assert relative_normal < 0  # Surfaces are approaching.
                    reconstructed = tuple(v - (1+e)*relative_normal*a for v,a in zip(velocity,n))
                    assert math.dist(reconstructed, s['outgoing']) < 1e-6
                    if mode != 'direct':
                        a, b, c = s['path']
                        vin = unit((b[0]-a[0], b[1]-a[1]))
                        vout = unit((c[0]-b[0], c[1]-b[1]))
                        assert math.dist((-vin[0], vin[1]), vout) < 1e-7
                        assert b[0] == (35 if mode == 'left' else 965)
                    count += 1
    try:
        plan_shot((500, 1000), (0, 0), 1000, 35, (480, 520, 0))
    except ValueError:
        pass
    else:
        raise AssertionError('Narrow goal accepted')
    print(f'PASS: {count} shot/collision cases and narrow-goal rejection')


if __name__ == '__main__':
    self_test()
