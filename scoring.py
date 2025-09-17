def atc_score(body_length, height_at_withers, chest_width, rump_angle):
    """
    Simple example scoring rules - points based on measurement thresholds
    Adjust thresholds and points as per domain expert inputs
    """
    score = 0

    # Example thresholds for body length (in cm)
    if body_length > 150:
        score += 30
    elif body_length > 140:
        score += 25
    elif body_length > 130:
        score += 20
    else:
        score += 10

    # Height at withers
    if height_at_withers > 120:
        score += 30
    elif height_at_withers > 110:
        score += 25
    elif height_at_withers > 100:
        score += 20
    else:
        score += 10

    # Chest width
    if chest_width > 50:
        score += 20
    elif chest_width > 40:
        score += 15
    else:
        score += 10

    # Rump angle (ideal range ~70-90 degrees)
    if 70 <= rump_angle <= 90:
        score += 20
    elif 60 <= rump_angle < 70 or 90 < rump_angle <= 100:
        score += 15
    else:
        score += 10

    return score
