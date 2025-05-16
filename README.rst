|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Эффективное агрегирование по меткам для задачи последовательностей событий
    :Тип научной работы: НИР
    :Автор: Боева Галина Леонидовна
    :Научный руководитель: к.ф-м.н, Алексей Алексеевич Зайцев

Abstract
========

Most user-related data can be represented as a sequence of events associated with a timestamp and a collection of categorical labels. For example, the purchased basket of goods and the time of buying fully characterize the event of the store visit. Anticipation of the label set for the future event called the problem of temporal sets prediction, holds significant value, especially in such high-stakes industries as finance and e-commerce. A fundamental challenge of this task is the joint consideration of the temporal nature of events and label relations within sets. The existing models fail to capture complex time and label dependencies due to ineffective representation of
historical information initially. We aim to address this shortcoming by presenting the framework with a specific way to aggregate the observed information into time- and set structure-aware views prior to transferring it into main architecture blocks. Our strong emphasis on input arrangement facilitates the subsequent efficient learning of label interactions. The proposed model is called Label-Attention NETwork, or LANET. We conducted experiments on four different datasets and made a comparison with four established models, including SOTA, in this area. The experimental results suggest that LANET provides significantly better quality than any other model, achieving an improvement up to 65% in terms of weighted F1 metric compared to the closest competitor. Moreover, we contemplate causal relationships between labels in our work, as well as a thorough study of LANET components’ influence on performance. We provide an implementation of LANET to encourage its wider usage.

Research publications
===============================
1. ECAI 2024

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.comintsystems/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/main.ipynb>`_.
