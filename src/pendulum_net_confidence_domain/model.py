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
measure of set... hmmm it could always shrink it, so it's better to do via distance in the real world?
"""
import torch
import torch.nn as nn
import numpy as np

from gen_data import load_data

AUTOENCODER_EPOCHS = 0
EPOCHS = 10

BITMAP_SPACE_DIMENSION = 4
CONVEX_SPACE_DIMENSION = 4  # * 2 if we count min and max

LEARNING_RATE = 1e-3

TRANSLATION_COEFFICIENT = 0
MEASURE_COEFFICIENT = 0.02
OFFSET_COEFFICIENT = 1
INVERSION_COEFFICIENT = 1
C_COEFFICIENT = 0

C_BUFFER = 0.05
MEASURE_SAMPLES = 0
PFRAMES = 24

INSTANCES_IN_BATCH = 32


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
        self.dense1 = dense(BITMAP_SPACE_DIMENSION, 4)
        self.dense2 = dense(4, 8)
        self.dense3 = dense(8, CONVEX_SPACE_DIMENSION)

    def forward(self, x):
        out_dense1 = self.dense1(x)
        out_dense2 = self.dense2(out_dense1)
        out_dense3 = self.dense3(out_dense2)
        return out_dense3


# from a SINGLE convex point
class FromConvex(nn.Module):

    def __init__(self):
        super(FromConvex, self).__init__()
        self.dense1 = dense(CONVEX_SPACE_DIMENSION, 8)
        self.dense2 = dense(8, 4)
        self.dense3 = dense(4, BITMAP_SPACE_DIMENSION)

    def forward(self, x):
        out_dense1 = self.dense1(x)
        out_dense2 = self.dense2(out_dense1)
        out_dense3 = self.dense3(out_dense2)
        return out_dense3


# step a BOX of convex points
class Step(nn.Module):

    def __init__(self):
        super(Step, self).__init__()
        self.dropout = nn.Dropout()
        self.dense1 = dense(CONVEX_SPACE_DIMENSION * 2 * PFRAMES + 1, 32)
        self.dense2 = dense(32, 32)
        self.dense3 = dense(32, 32)
        self.dense4 = dense(32, 2 * CONVEX_SPACE_DIMENSION)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse_loss = nn.MSELoss()

    for e in range(EPOCHS + AUTOENCODER_EPOCHS):

        for b in range(0, len(data), INSTANCES_IN_BATCH):
            translation_loss = torch.scalar_tensor(0)
            measure_loss = torch.scalar_tensor(0)
            c_loss = torch.scalar_tensor(0)
            offset_loss = torch.scalar_tensor(0)
            inversion_loss = torch.scalar_tensor(0)

            contained_vector = {}

            extension = PFRAMES + max(1, 0 * int(e / EPOCHS * (data.shape[1] - PFRAMES)))
            batch = data[b:min(len(data), b + INSTANCES_IN_BATCH), :extension]

            rand = batch  # seems like batch is better than using random?
            convex = model.to_convex(rand)
            bitmap = model.from_convex(convex)

            translation_loss += mse_loss(bitmap, rand) * TRANSLATION_COEFFICIENT

            if e >= AUTOENCODER_EPOCHS:
                # c = torch.rand(1)
                c = torch.tensor([1])

                convex = batch[:, :PFRAMES]  # model.to_convex(batch[:, 0])
                frame = convex[:, 0]

                contained_vector.setdefault(float(c), 0)

                for i in range(PFRAMES, batch.shape[1]):
                    flat_prev = torch.concat((convex, convex), dim=-1)
                    flat_prev = torch.flatten(flat_prev, 1, 2)
                    in_vec = torch.concat((flat_prev, torch.full((batch.shape[0], 1), 1)), dim=1)
                    frame = model.step(in_vec)

                    shift = convex[:, 1: PFRAMES]
                    convex = torch.concat((shift, torch.unsqueeze(frame[:, :CONVEX_SPACE_DIMENSION], dim=1)), dim=1)

                    offset_loss += mse_loss(batch[:, i], frame[:, :CONVEX_SPACE_DIMENSION])
                    if torch.rand(1) < 0.0001:
                        print("Break")
                """
                distances = torch.zeros((batch.shape[0], batch.shape[1] - 1))
                for i in range(1, max(2, int(e * batch.shape[1] / (EPOCHS + AUTOENCODER_EPOCHS)))):
                    convex = model.step(torch.concat([frame, frame, torch.full((batch.shape[0], 1), float(c))], dim=1))
                    frame = batch[:, i]
                    convex_frame = model.to_convex(frame)

                    # entirely contained within the convex set
                    min_dimension = convex[:, :CONVEX_SPACE_DIMENSION]
                    width = convex[:, CONVEX_SPACE_DIMENSION:]
                    max_dimension = min_dimension + width
                    avg_dimension = min_dimension + width / 2
                    offset_loss += OFFSET_COEFFICIENT * (batch.shape[1] - i) / batch.shape[1] * \
                        mse_loss(min_dimension, frame) / (batch.shape[1] - 1)

                    inversion_loss = INVERSION_COEFFICIENT * torch.sum(torch.max(torch.zeros(1), C_BUFFER - width))

                    inside = torch.logical_and(torch.all(convex_frame >= min_dimension, dim=1),
                                               torch.all(convex_frame <= max_dimension, dim=1))

                    # loss of size of S by taking some random sample of points
                    samples = torch.rand((MEASURE_SAMPLES, batch.shape[0], CONVEX_SPACE_DIMENSION)) * width
                    samples += min_dimension
                    samples = model.from_convex(samples)  # [batch size, measure samples, bitmap space]
                    measure = 0
                    for j, sample in enumerate(samples):
                        for sample2 in samples[j:]:
                            measure += mse_loss(sample, sample2)
                    # # maybe try MEASURE_SAMPLES * (MEASURE_SAMPLES - 1)?
                    measure_loss += measure * MEASURE_COEFFICIENT

                    contained_vector[float(c)] += torch.sum(inside)

                    # so for all axes that it's not currently aligned,
                    # take the manhattan distance of whichever is closer
                    clipped_width = torch.max(width, torch.tensor([C_BUFFER]))
                    pivot = torch.max(convex_frame - avg_dimension, clipped_width)
                    full = torch.log(pivot) - torch.log(clipped_width + torch.tensor([C_BUFFER]))
                    distances[:, i - 1] = torch.mean(full, dim=1)

                # loss of making sure C is respected at every frame
                torch.sort(distances, dim=0)
                c_loss += torch.mean(distances[:int(torch.ceil(c * len(distances)))]) * C_COEFFICIENT
                """
            # have some diversity in training data
            loss = offset_loss + inversion_loss  # + c_loss + measure_loss + translation_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b % 3 == 0:
                print("epoch", e,
                      "batch", b,
                      "loss", float(loss),
                      "offset-loss", float(offset_loss),
                      "c-loss", float(c_loss),
                      "inversion-loss", float(inversion_loss),
                      "trans-loss", float(translation_loss.data),
                      "measure-loss", float(measure_loss.data),
                      "last contained", contained_vector
                      )

                # print(model.from_convex(model.to_convex(torch.tensor([0.1, 0.3, 0, 0]))))
                # if e >= AUTOENCODER_EPOCHS:
                #    print("min", min_dimension, "example", convex_frame, "max", max_dimension)

    torch.save(model, "../../models/pendulum_net.pt")


def load_cached_model():
    return torch.load("../../models/pendulum_net.pt")


def model_from_parameters(parameters):
    m1 = Model()
    for param, src in zip(m1.parameters(), parameters):
        param.data = src.data

    return m1


if __name__ == '__main__':
    gen_trained_model()
    # manim runs from a different path, which means that pickle acts strangely...
    # so just save the parameters and input it ourselves
    m = load_cached_model()
    params = list(m.parameters())
    torch.save(params, "../../models/pendulum_net_parameters.pt")
