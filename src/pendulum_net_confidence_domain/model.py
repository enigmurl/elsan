
"""
ok planning
so we have autoencoder with convex space
then we have rnn like neural net actually run

For here, dense layers are fine

ok so LOSS: Say we have a current batch, and we're running it at C
then take the amount of predictions that actually fell inside the bounding box
if it's less than C, we need to prioritize reducing the summed distance to the outside points
if it's greater than c, we need to prioritize reduce the measure of the set (however we do that...)

we also need a general loss for the translater based on how far off it's getting from retranslation

so loss = ∑(activated distance of outside points as a function of c) +
            measure of set (likely via mse for some sample of points in real world) +
            autoencoder translation loss
measure of set... hmmmmm it could always shrink it, so it's better to do via distance in the real world?
"""
import torch
import torch.nn as nn
import numpy as np

from gen_data import load_data

EPOCHS = 3
CONVEX_SPACE_DIMENSION = 8  # * 2 if we count min and max

TRANSLATION_COEFFICIENT = 1
MEASURE_COEFFICIENT = 1
C_COEFFICIENT = 1

C_SAMPLES = 10
MEASURE_SAMPLES = 10


def dense(in_feature, out_features):
    layer = nn.Sequential(
        nn.Linear(in_feature, out_features),
        nn.LeakyReLU()
    )
    return layer


# to a SINGLE convex point
class ToConvex(nn.Module):

    def __init__(self):
        super(ToConvex, self).__init__()
        self.dense1 = dense(2, 4)
        self.dense2 = dense(4, 8)
        self.dense3 = dense(8, 8)

    def forward(self, x):
        out_dense1 = self.dense1(x)
        out_dense2 = self.dense2(out_dense1)
        out_dense3 = self.dense3(out_dense2)
        return out_dense3


# from a SINGLE convex point
class FromConvex(nn.Module):

    def __init__(self):
        super(FromConvex, self).__init__()
        self.dense1 = dense(8, 8)
        self.dense2 = dense(8, 4)
        self.dense3 = nn.Linear(4, 2)

    def forward(self, x):
        out_dense1 = self.dense1(x)
        out_dense2 = self.dense2(out_dense1)
        out_dense3 = self.dense3(out_dense2)
        return out_dense3


# step a BOX of convex points
class Step(nn.Module):

    def __init__(self):
        super(Step, self).__init__()
        self.dense1 = dense(CONVEX_SPACE_DIMENSION * 2 + 1, 16)
        self.dense2 = dense(16, 16)
        self.dense3 = dense(16, 16)
        self.dense4 = dense(16, 16)

    def forward(self, x):
        out_dense1 = self.dense1(x)
        out_dense2 = self.dense2(out_dense1)
        out_dense3 = self.dense3(out_dense2)
        out_dense4 = self.dense4(out_dense3)
        return out_dense4


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.to_convex = ToConvex()
        self.step = Step()
        self.from_convex = FromConvex()

    def forward(self, x):
        convex = self.to_convex(x)
        return self.step(torch.concat((convex, convex)))


def rand_directions(samples, dimension):
    """
    Random points on a hypersphere embedded in R^dimension
    Somewhat, useful to track approximately how a single point moves
    """
    directions = np.random.normal(0, 1, (samples, dimension))
    return torch.tensor(directions / np.linalg.norm(directions, axis=1))


def gen_trained_model():
    data = load_data()

    model = Model()

    optimizer = torch.optim.Adam(model.parameters())
    mse_loss = nn.MSELoss()

    for e in range(EPOCHS):
        for b, batch in enumerate(data):
            loss = torch.scalar_tensor(0)

            # loss of making sure that each element in batch can be translated properly
            # in tf_net final version, we can optimize the for loops into one??
            for frame in batch:
                convex = model.to_convex(frame)
                bitmap = model.from_convex(convex)

                translation = mse_loss(bitmap, frame)
                loss += translation * TRANSLATION_COEFFICIENT / len(batch)

            # in tf_net, probably want to have a less uniform batch
            for c in torch.linspace(0, 1, C_SAMPLES):
                convex = model.to_convex(batch[0])
                convex = torch.concat((convex, convex))

                contained = 0
                distances = torch.scalar_tensor(0)

                for frame in batch[1:]:
                    convex = model.step(torch.concat([convex, torch.tensor([c])]))

                    convex_frame = model.to_convex(frame)

                    # entirely contained within the convex set
                    min_dimension = convex[:CONVEX_SPACE_DIMENSION]
                    max_dimension = convex[CONVEX_SPACE_DIMENSION:]

                    inside = torch.all(convex_frame >= min_dimension) and torch.all(convex_frame <= max_dimension)

                    # loss of size of S by taking some random sample of points
                    samples = torch.rand((MEASURE_SAMPLES, CONVEX_SPACE_DIMENSION)) * (max_dimension - min_dimension)
                    samples += min_dimension
                    samples = list(map(lambda s: model.from_convex(s), samples))

                    measure = 0
                    for i, sample in enumerate(samples):
                        for sample2 in samples[i:]:
                            measure += mse_loss(sample, sample2)
                    loss += measure * MEASURE_COEFFICIENT / (len(batch) * MEASURE_SAMPLES * (MEASURE_SAMPLES - 1) / 2)

                    if inside:
                        contained += 1
                    else:
                        # so for all axes that it's not currently aligned,
                        # take the manhattan distance of whichever is closer
                        full = torch.zeros(CONVEX_SPACE_DIMENSION)
                        full[convex_frame < min_dimension] = (min_dimension - convex_frame)[convex_frame < min_dimension]
                        full[convex_frame > max_dimension] = (convex_frame - max_dimension)[convex_frame > max_dimension]
                        distances += torch.sigmoid(torch.sum(full)) / len(batch) ** 2

                # loss of making sure C is respected
                if contained / len(batch) >= c:
                    c_loss = 0
                else:
                    c_loss = distances

                loss += c_loss * C_COEFFICIENT

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % (1 << 8) == 0:
                print(b, "loss", loss, "last contained", contained)

    print("Finish")


if __name__ == '__main__':
    gen_trained_model()
