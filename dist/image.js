import sharp from "sharp";
/** @internal Verifies the invariant established by Sharp's ensureAlpha pipeline. */
export function assertFourDecodedChannels(channels) {
    if (channels !== 4) {
        throw new Error(`Expected four decoded channels, received ${channels}`);
    }
}
export async function decodeImage(input) {
    const source = typeof input === "string" ? input : Buffer.from(input);
    const { data, info } = await sharp(source)
        .ensureAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });
    assertFourDecodedChannels(info.channels);
    return {
        data,
        width: info.width,
        height: info.height,
        channels: 4,
    };
}
export async function resizeRawImage(image, height) {
    const width = Math.round((image.width / image.height) * height);
    const { data, info } = await sharp(Buffer.from(image.data), {
        raw: { width: image.width, height: image.height, channels: image.channels },
    })
        .resize({ width, height, kernel: "nearest" })
        .raw()
        .toBuffer({ resolveWithObject: true });
    return { data, width: info.width, height: info.height, channels: 4 };
}
//# sourceMappingURL=image.js.map