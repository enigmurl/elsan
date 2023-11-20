# Ensemble Linear Sum Assignment Network (ELSAN)

The idea of this project was to see if we can use the Linear Sum Assignment problem as a metric between distributions, thereby allowing models that can encode entire distributions, not just single results. We specifically focused on chaotic spatiotemporal systems, in particular turbulent fluid flow. It unfortunately did not perform very well since this approach is somewhat Monte-Carlo in nature, and is thus outperforemd by Variational Autoencoders.

Nevertheless, we believe that the linear sum assignment function is under utilized. We also show how square root decomposition can be used to speed up inference and training of spatiotemporal models. Finally, in the paper, we show LSA can be taken to be a metric.
