from trans_encoder import *


class FeatFuseLater(nn.Module):

    def __init__(self, tasks):
        super(FeatFuseLater, self).__init__()
        self.decoder = TransEncoder(inc=512, outc=512, dropout=0.3, nheads=4, nlayer=4)

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
        )



        self.dropout = nn.Dropout(p=0.3)

        if tasks == 'EXPR':
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 8),
                nn.Sigmoid()
            )
        elif tasks == 'AU':
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 8),
                nn.Sigmoid()
            )
        elif tasks == 'VA':
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 8),
                nn.Sigmoid()
            )
    def forward(self, x):
        # img_feat (batch=16, dim=512)
        # audio_feat (batch=16, dim=256)
        # video_feat (batch=16, dim=768)
        img_feat, audio_feat, video_feat = x
        img_feat = img_feat.unsqueeze(1)
        feat = torch.cat([audio_feat, video_feat], dim=1)
        print(feat.shape)
        aout = self.fc1(feat)
        aout = aout.unsqueeze(1)
        print(aout.shape)
        vout = self.fc2(feat)
        vout = vout.unsqueeze(1)
        feat = torch.cat([aout, vout, img_feat], dim=1)
        feat = feat.transpose(1, 2)
        print(feat.shape)
        out = self.decoder(feat)
        out = out[2]



        out = self.classifier(out)
        print('out', out.shape)


        return img_feat, vout, aout

if __name__ == '__main__':
    model = FeatFuseLater('EXPR')
    img_feat = torch.randn(16, 512)
    audio_feat = torch.randn(16, 256)
    video_feat = torch.randn(16, 768)
    x = (img_feat, audio_feat, video_feat)

    model(x)
