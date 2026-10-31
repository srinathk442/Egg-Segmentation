"""Build a labelled contact sheet of all dataset images (quick visual EDA)."""
import sys, glob, os, cv2, numpy as np
root, out = sys.argv[1], sys.argv[2]
paths = sorted(glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True))
t, cols = 160, 12
rows = (len(paths) + cols - 1) // cols
sheet = np.full((rows * t, cols * t, 3), 255, np.uint8)
sizes = set()
for i, p in enumerate(paths):
    im = cv2.imread(p); sizes.add(im.shape)
    im = cv2.resize(im, (t, t))
    name = os.path.basename(p).split("_jpg")[0].replace("not_damaged_", "nd")
    cv2.putText(im, name, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    r, c = divmod(i, cols); sheet[r*t:(r+1)*t, c*t:(c+1)*t] = im
cv2.imwrite(out, sheet); print(len(paths), "images, shapes:", sizes)
