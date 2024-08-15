kernel = shoemark_feature_maps[78]
image = shoeprint_feature_maps[78]

kernel_height, kernel_width = kernel.shape
image_height, image_width = image.shape

pad_width = image_width + kernel_width
pad_height = image_height + kernel_height

image_width_pad = (kernel_width - 1) // 2
image_width_pad_extra = (kernel_width - 1) % 2
image_height_pad = (kernel_height - 1) // 2
image_height_pad_extra = (kernel_height - 1) % 2

padded_image = np.pad(
    image,
    pad_width=(
        (image_height_pad, image_height_pad + image_height_pad_extra),
        (image_width_pad, image_width_pad + image_width_pad_extra),
    ),
    mode="constant",
    constant_values=np.nan,
)

# ncc_image = image.copy()
ncc_image = np.zeros(image.shape)


def normxcorr(image1, image2):
    assert image1.shape == image2.shape, "Images must have the same shape"

    image1 = image1.copy()
    image1 = np.nan_to_num(image1)

    image2 = image2.copy()
    image2 = np.nan_to_num(image2)

    mean1 = np.mean(image1)
    mean2 = np.mean(image2)

    image1 -= mean1
    image2 -= mean2

    xcorr = np.sum(image1 * image2)

    std1 = np.std(image1)
    std2 = np.std(image2)

    zncc = xcorr / (std1 * std2 * image1.size)

    return zncc


def pad_kernel(image_row, image_column):
    left_pad = image_column
    right_pad = image_width - image_column - 1

    top_pad = image_row
    bottom_pad = image_height - image_row - 1

    return np.pad(
        kernel.copy(),
        pad_width=(
            (top_pad, bottom_pad),
            (left_pad, right_pad),
        ),
        mode="constant",
        constant_values=np.nan,
    )


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.axis("off")
ax2.axis("off")

img1 = ax1.imshow(padded_image)
img2 = ax2.imshow(ncc_image, norm=plt.Normalize(vmin=0, vmax=0), cmap="magma")


def make_frame(image_pixel):
    image_row, image_column = np.unravel_index(image_pixel, image.shape)

    padded_kernel = pad_kernel(image_row, image_column)

    mask = ~np.isnan(padded_kernel)
    joined_img = padded_image.copy()
    joined_img[mask] = padded_kernel[mask]

    img1.set_data(joined_img)

    padded_image_slice = padded_image[mask].reshape(kernel.shape)

    zncc = normxcorr(kernel, padded_image_slice)

    # print(zncc)

    ncc_image[image_row, image_column] = zncc

    img2.set_data(ncc_image)
    img2.set_norm(Normalize(vmin=ncc_image.min(), vmax=ncc_image.max()))

    return img1, img2


ani = FuncAnimation(fig, make_frame, frames=image.size, interval=1, blit=True)

plt.subplots_adjust(wspace=0)
plt.tight_layout()

writer = FFMpegWriter(fps=60, bitrate=1800)
writer.ffmpeg_params = ["-filter:v", "setpts=0.5*PTS"]
ani.save("animation.mp4", writer=writer)
