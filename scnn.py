import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialMessagePassing(nn.Module):
    def __init__(self, channels, direction='horizontal'):
        super(SpatialMessagePassing, self).__init__()
        self.direction = direction
        self.conv = nn.Conv2d(channels, channels, kernel_size=9, stride=1, padding=4, groups=channels, bias=False)

    def forward(self, x):
        if self.direction == 'horizontal':
            x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
            x = self.conv(x.permute(0, 3, 1, 2))
            x = x.permute(0, 2, 3, 1)  # NHWC -> NCHW
        elif self.direction == 'vertical':
            x = x
            x = self.conv(x)
        return x


class SCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super(SCNN, self).__init__()

        # Backbone (like VGG16 conv blocks)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # SCNN spatial message passing
        self.smp_vertical = SpatialMessagePassing(512, direction='vertical')
        self.smp_horizontal = SpatialMessagePassing(512, direction='horizontal')

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x):
        x = self.backbone(x)

        # Spatial Message Passing
        x = self.smp_vertical(x)
        x = self.smp_horizontal(x)

        out = self.decoder(x)
        out = F.interpolate(out, size=(x.shape[2]*8, x.shape[3]*8), mode='bilinear', align_corners=False)
        return out


if __name__ == "__main__":
    model = SCNN()
    dummy_input = torch.randn(1, 3, 288, 800)  # typical CULane input size
    out = model(dummy_input)
    print("Output shape:", out.shape)  # should be [1, num_classes, 288, 800] or upsampled
