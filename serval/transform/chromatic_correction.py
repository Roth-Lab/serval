from collections import defaultdict

import itertools

import numpy as np
import scipy.signal as signal
import skimage
import copy

from serval.transform import ImageTransform


class ChromaticCorrectionImageTransform(ImageTransform):
    def __init__(self, bit_to_color_map, min_area=5, filter_outliers=True):
        self.bit_to_color_map = copy.deepcopy(bit_to_color_map)

        self.min_area = min_area

        self.filter_outliers = filter_outliers

        self.ref_color = min(bit_to_color_map.values())

        colors = set(bit_to_color_map.values())

        colors.remove(self.ref_color)

        self.img_transforms = {
            c: skimage.transform.SimilarityTransform() for c in colors
        }

    @property
    def params(self):
        return self.img_transforms

    @params.setter
    def params(self, x):
        self.img_transforms = x

    def get_update_params(self, local_params):
        color_displacements = defaultdict(list)

        for x in local_params:
            for c in x:
                color_displacements[c].extend(x[c])

        img_transforms = {}

        for color, offsets in color_displacements.items():
            if self.filter_outliers:
                offsets = self._filter_offsets(offsets)

            t = skimage.transform.SimilarityTransform()

            try:
                t.estimate(
                    np.array([x[0] for x in offsets]),
                    np.array([x[0] + x[1] for x in offsets]),
                )
            except np.linalg.LinAlgError:
                print(f"Warning: SVD did not converge for color {color}.")
                continue  # Or handle in another way

            # TODO: This only makes sense if we compute after images receive previous transform. Is that necessary?
            img_transforms[color] = self.img_transforms[color] + t

        return img_transforms

    def get_local_update_params(self, decoded, imgs):
        imgs = self.transform(imgs)

        return self.get_offsets(decoded, imgs)

    def get_offsets(self, decoded, imgs, debug=False):
        color_displacements = defaultdict(list)

        x_size = imgs.imgs.shape[2]

        y_size = imgs.imgs.shape[1]

        imgs_f = imgs.imgs.astype(float)

        spots_df = decoded.spots

        spots_df = spots_df[spots_df["area"] >= self.min_area]

        trace = []

        for _, row in spots_df.iterrows():
            barcode = decoded.codebook.get_barcode(row["target"])

            on_bits = [i for i, b in enumerate(barcode) if b == 1]

            x, y = row[["x", "y"]]

            if is_near_border((x, y), (x_size, y_size)):
                continue

            refined_positions = np.array(
                [refine_position(imgs_f[i], x, y) for i in on_bits]
            )

            for i, j in itertools.combinations(range(len(on_bits)), 2):
                c_1 = self.bit_to_color_map[on_bits[i]]

                c_2 = self.bit_to_color_map[on_bits[j]]

                if c_1 == c_2:
                    continue

                elif (c_1 != self.ref_color) and (c_2 != self.ref_color):
                    continue

                elif c_1 == self.ref_color:
                    color_displacements[c_2].append(
                        [
                            np.array([x, y]),
                            refined_positions[j] - refined_positions[i],
                        ]
                    )

                    ref = c_1
                    ref_x, ref_y = refined_positions[i]
                    ref_bit = on_bits[i]
                    alt = c_2
                    alt_x, alt_y = refined_positions[j]
                    alt_bit = on_bits[j]

                elif c_2 == self.ref_color:
                    color_displacements[c_1].append(
                        [
                            np.array([x, y]),
                            refined_positions[i] - refined_positions[j],
                        ]
                    )

                    ref = c_2
                    ref_x, ref_y = refined_positions[j]
                    ref_bit = on_bits[j]
                    alt = c_1
                    alt_x, alt_y = refined_positions[i]
                    alt_bit = on_bits[i]

                else:
                    print(c_1, c_2, self.ref_color)
                    raise

                trace.append(
                    {
                        "fov": imgs.fov,
                        "z": imgs.z,
                        "barcode_id": decoded.codebook.get_target_id(row["target"]),
                        "target": row["target"],
                        "ref": ref,
                        "alt": alt,
                        "ref_bit": ref_bit,
                        "alt_bit": alt_bit,
                        "x": row["x"],
                        "y": row["y"],
                        "ref_x": ref_x,
                        "ref_y": ref_y,
                        "alt_x": alt_x,
                        "alt_y": alt_y,
                    }
                )

        if debug:
            return color_displacements, trace

        else:
            return color_displacements

    def _transform(self, img, bit, fov, z):
        c = self.bit_to_color_map[bit]
        
        #print(f"bit_to_color_map: {self.bit_to_color_map}")

        if c == self.ref_color:
            return img

        else:
            #print(f"img_transforms: {self.img_transforms}")
            return skimage.transform.warp(
                img, self.img_transforms[c], preserve_range=True
            )

    def _filter_offsets(self, offsets):
        keep_idxs = [
            i
            for i, x in enumerate(offsets)
            if not any(np.isnan(x[1])) and not any(np.isinf(x[1]))
        ]

        offsets = [offsets[i] for i in keep_idxs]

        diff_x = np.array([x[1][0] for x in offsets])

        diff_y = np.array([x[1][1] for x in offsets])

        q_x_l = np.quantile(diff_x, 0.01)

        q_x_h = np.quantile(diff_x, 0.99)

        q_y_l = np.quantile(diff_y, 0.01)

        q_y_h = np.quantile(diff_y, 0.99)

        keep_idxs = [
            i
            for i, x in enumerate(offsets)
            if (x[1][0] >= q_x_l)
            and (x[1][0] <= q_x_h)
            and (x[1][1] >= q_y_l)
            and (x[1][1] <= q_y_h)
        ]

        return [offsets[i] for i in keep_idxs]


def is_near_border(coords, sizes, border_size=10):
    near_border = False

    for i in range(2):
        if coords[i] <= border_size:
            near_border = True

        if (sizes[i] - coords[i]) <= border_size:
            near_border = True

    return near_border


def lsradialcenterfit(m, b, w):
    wm2p1 = w / (m * m + 1)
    sw = np.sum(wm2p1)
    smmw = np.sum(m * m * wm2p1)
    smw = np.sum(m * wm2p1)
    smbw = np.sum(m * b * wm2p1)
    sbw = np.sum(b * wm2p1)
    det = smw * smw - smmw * sw
    xc = (smbw * sw - smw * sbw) / det
    yc = (smbw * smw - smmw * sbw) / det

    return xc, yc


def radial_center(imageIn):
    """Determine the center of the object in imageIn using radial-symmetry-based
    particle localization.

    Adapted from Raghuveer, Nature Methods, 2012
    """
    Ny, Nx = imageIn.shape
    xm_onerow = np.arange(-(Nx - 1) / 2.0 + 0.5, (Nx) / 2.0 - 0.5)
    xm = np.tile(xm_onerow, (Ny - 1, 1))
    ym_onecol = [np.arange(-(Nx - 1) / 2.0 + 0.5, (Nx) / 2.0 - 0.5)]
    ym = np.tile(ym_onecol, (Nx - 1, 1)).transpose()

    imageIn = imageIn.astype(float)

    dIdu = imageIn[0 : Ny - 1, 1:Nx] - imageIn[1:Ny, 0 : Nx - 1]
    dIdv = imageIn[0 : Ny - 1, 0 : Nx - 1] - imageIn[1:Ny, 1:Nx]

    h = np.ones((3, 3)) / 9
    fdu = signal.convolve2d(dIdu, h, "same")
    fdv = signal.convolve2d(dIdv, h, "same")
    dImag2 = np.multiply(fdu, fdu) + np.multiply(fdv, fdv)

    # TODO: Restructure this code to get rid of the divide by zero warning
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.divide(-(fdv + fdu), (fdu - fdv + 1e-8))

    if np.any(np.isnan(m)):
        unsmoothm = np.divide(dIdv + dIdu, dIdu - dIdv + 1e-8)
        m[np.isnan(m)] = unsmoothm[np.isnan(m)]

    if np.any(np.isnan(m)):
        m[np.isnan(m)] = 0

    if np.any(np.isinf(m)):
        if ~np.all(np.isinf(m)):
            m[np.isinf(m)] = 10 * np.max(m[~np.isinf(m)])
        else:
            m = np.divide((dIdv + dIdu), (dIdu - dIdv))

    b = ym - np.multiply(m, xm)

    sdI2 = np.sum(dImag2)
    xcentroid = np.sum(np.sum(np.multiply(dImag2, xm))) / sdI2
    ycentroid = np.sum(np.multiply(dImag2, ym)) / sdI2
    w = np.divide(
        dImag2,
        np.sqrt(
            (xm - xcentroid) * (xm - xcentroid) + (ym - ycentroid) * (ym - ycentroid)
        ),
    )

    xc, yc = lsradialcenterfit(m, b, w)

    xc = xc + (Nx + 1) / 2.0
    yc = yc + (Ny + 1) / 2.0

    return xc, yc


def refine_position(image, x, y, cropSize=4):
    # Note: This is slightly different than the original MERLIN code which wrapped the indices in int()
    subImage = image[
        round(y) + 2 - cropSize : round(y) + cropSize,
        round(x) + 2 - cropSize : round(x) + cropSize,
    ]
    return radial_center(subImage)


# def refine_position(img, x, y, crop_size=4):
#     x_i = round(x)
#     y_i = round(y)
#     try:
#         x_s = slice(x_i - crop_size, x_i + crop_size + 1)
#         y_s = slice(y_i - crop_size, y_i + crop_size + 1)
#         sub_img = img[y_s, x_s]
#         x_c, y_c = radial_center(sub_img)
#         x_fit = x_i + x_c - crop_size
#         y_fit = y_i + y_c - crop_size
#     # TODO: This is mainly to catch crop size issues but could hide other errors
#     except ValueError:
#         x_fit = x
#         y_fit = y
#     return x_fit, y_fit
