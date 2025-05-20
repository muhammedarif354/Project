# Currency note authentication

import cv2

# Load the images
img1 = cv2.imread('realcurrency.png', 0)  # Template (genuine)
img2 = cv2.imread('fakecurrency.png', 0)  # Note to verify

# Convert grayscale to BGR for annotation
img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=1000)

# Find the keypoints and descriptors
kp1, des1 = orb.detectAndCompute(img1_color, None)
kp2, des2 = orb.detectAndCompute(img2_color, None)

# Create Brute Force Matcher and match descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Count good matches based on distance threshold
good_matches = [m for m in matches if m.distance < 60]

# Decision threshold
if len(good_matches) > 25:
    verdict = "✅ GENUINE"
    color = (0, 255, 0)
else:
    verdict = "❌ FAKE or TAMPERED"
    color = (0, 0, 255)

# Add decision text on both images
cv2.putText(img1_color, "GENUINE TEMPLATE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
cv2.putText(img2_color, f"NOTE UNDER TEST: {verdict}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# Show images in separate windows
cv2.imshow("Real Currency (Template)", img1_color)
cv2.imshow("Currency Under Test", img2_color)

# Optional: Show matches in a separate window too
match_img = cv2.drawMatches(img1_color, kp1, img2_color, kp2, matches[:30], None, flags=2)
cv2.imshow("Keypoint Matches", match_img)

cv2.waitKey(0)
cv2.destroyAllWindows()