"""
class ResBlock(Layer):
    def __init__(self, channels, **kwargs):
        super().__init__()
        self.channels = channels
        self.stride = kwargs.pop("stride", 1)
        time_dist = kwargs.get("time_dist", False)

        TD = TimeDistributed if time_dist else (lambda x: x)

        if self.stride > 1:
            self.pool = TD(AveragePooling2D(
                pool_size=(self.stride, self.stride)))
        else:
            self.pool = lambda x: x
        self.proj = TD(Conv2D(self.channels, kernel_size=(1, 1)))

        self.conv_block_1 = ConvBlock(channels, stride=self.stride, **kwargs)
        self.conv_block_2 = ConvBlock(channels, activation='leakyrelu', **kwargs)
        self.add = Add()

    def call(self, x):
        x_in = self.pool(x)
        in_channels = int(x.shape[-1])
        if in_channels != self.channels:
            x_in = self.proj(x_in)

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        return self.add([x, x_in])


class GRUResBlock(ResBlock):
    def __init__(self, channels, final_activation='sigmoid', **kwargs):
        super().__init__(channels, **kwargs)
        self.final_act = Activation(final_activation)

    def call(self, x):
        x = super().call(x)
        return self.final_act(x)

class CustomGateGRU(Layer):
    def __init__(self,
        update_gate=None, reset_gate=None, output_gate=None,
        return_sequences=False, time_steps=1,
        **kwargs):

        super().__init__(**kwargs)

        self.update_gate = update_gate
        self.reset_gate = reset_gate
        self.output_gate = output_gate
        self.return_sequences = return_sequences
        self.time_steps = time_steps

    def call(self, inputs):
        (xt,h) = inputs

        h_all = []
        for t in range(self.time_steps):
            x = xt[:,t,...]
            xh = tf.concat((x,h), axis=-1)
            z = self.update_gate(xh)
            r = self.reset_gate(xh)
            o = self.output_gate(tf.concat((x,r*h), axis=-1))
            h = z*h + (1-z)*tf.math.tanh(o)
            if self.return_sequences:
                h_all.append(h)

        return tf.stack(h_all,axis=1) if self.return_sequences else h


class ConvGRU(Layer):
    def __init__(self, channels, conv_size=(3,3),
        return_sequences=False, time_steps=1,
        **kwargs):

        super().__init__(**kwargs)

        self.update_gate = Conv2D(channels, conv_size, activation='sigmoid',
            padding='same')
        self.reset_gate = Conv2D(channels, conv_size, activation='sigmoid',
            padding='same')
        self.output_gate = Conv2D(channels, conv_size, padding='same')

        self.return_sequences = return_sequences
        self.time_steps = time_steps

    def call(self, inputs):
        (xt,h) = inputs

        h_all = []
        for t in range(self.time_steps):
            x = xt[:,t,...]
            xh = tf.concat((x,h), axis=-1)
            z = self.update_gate(xh)
            r = self.reset_gate(xh)
            o = self.output_gate(tf.concat((x,r*h), axis=-1))
            h = z*h + (1-z)*tf.math.tanh(o)
            if self.return_sequences:
                h_all.append(h)

        return tf.stack(h_all,axis=1) if self.return_sequences else h


class ResGRU(ConvGRU):
    def __init__(self, channels, conv_size=(3,3),
        return_sequences=False, time_steps=1,
        **kwargs):

        super(ConvGRU, self).__init__(**kwargs)

        self.update_gate = GRUResBlock(channels, conv_size=conv_size,
            final_activation='sigmoid', padding='same')
        self.reset_gate = GRUResBlock(channels, conv_size=conv_size,
            final_activation='sigmoid', padding='same')
        self.output_gate = GRUResBlock(channels, conv_size=conv_size,
            final_activation='linear', padding='same')

        self.return_sequences = return_sequences
        self.time_steps = time_steps
"""
