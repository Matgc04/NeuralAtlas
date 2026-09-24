/* NeuralAtlas glossary — the reference layer behind every selector.
 *
 * Every entry fills the SAME slots, and the panel draws those slots in the same
 * place every time: what it is, what makes it different from the rest, and a
 * fixed row of facts. Switching entries swaps the values and moves nothing —
 * the way a telemetry readout changes numbers without changing its dials.
 *
 * That constraint is the schema. `facts` is an ordered list of [label, value],
 * and within a kind every entry MUST carry the same labels in the same order,
 * or the panel will jump when you switch. FACT_KEYS below is the contract.
 *
 * Method ids carry the segmentation in parentheses ("Lime (SLIC)"), so lookup
 * strips it and falls back to the base algorithm.
 */

const CAPTUM = 'https://captum.ai/api/';

// The fixed dial faces, per kind. Same labels, same order, every entry.
export const FACT_KEYS = {
  model: ['Family', 'Params', 'Year', 'Depth', 'Read-out'],
  dataset: ['Images', 'Classes', 'Source', 'Preprocessing'],
  method: ['Family', 'Needs', 'Cost', 'Resolution', 'Limit'],
  metric: ['Measures', 'Better', 'Removes', 'Limit'],
};

export const WIKI_SECTIONS = [
  {
    kind: 'model',
    label: 'Models',
    entries: [
      {
        id: 'convnext_tiny',
        title: 'ConvNeXt Tiny',
        tags: ['CNN', '2022'],
        summary: 'A convolutional network with four stages of residual blocks.',
        differs: 'Uses 7x7 depthwise convolutions, layer normalization and GELU activations.',
        facts: [
          ['Family', 'CNN, ConvNeXt'],
          ['Params', '28.6M'],
          ['Year', '2022'],
          ['Depth', '18 residual blocks'],
          ['Read-out', 'Last depthwise convolution'],
        ],
        links: [
          { label: 'torchvision: ConvNeXt Tiny', href: 'https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html' },
        ],
      },
      {
        id: 'efficientnet_b4',
        title: 'EfficientNet-B4',
        tags: ['CNN', 'Compound scaling'],
        summary: 'A larger EfficientNet variant that scales network depth, width and training resolution together.',
        differs: 'Uses more and wider inverted residual blocks than B0, with squeeze-and-excitation channel weighting.',
        facts: [
          ['Family', 'CNN, EfficientNet'],
          ['Params', '19.3M'],
          ['Year', '2019'],
          ['Depth', '32 inverted residual blocks'],
          ['Read-out', 'Last conv block'],
        ],
        links: [
          { label: 'torchvision: EfficientNet-B4', href: 'https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b4.html' },
        ],
      },
      {
        id: 'inception_v3',
        title: 'Inception v3',
        tags: ['CNN', 'Multi-branch'],
        summary: 'A CNN that combines parallel convolution and pooling branches in Inception modules.',
        differs: 'Factorizes larger convolutions into smaller or asymmetric kernels and concatenates features from parallel branches.',
        facts: [
          ['Family', 'CNN, Inception'],
          ['Params', '27.2M including auxiliary classifier'],
          ['Year', '2015'],
          ['Depth', '11 Inception modules'],
          ['Read-out', 'Last convolution in Mixed_7c'],
        ],
        links: [
          { label: 'torchvision: Inception v3', href: 'https://docs.pytorch.org/vision/main/models/generated/torchvision.models.inception_v3.html' },
        ],
      },
      {
        id: 'mobilenet_v2',
        title: 'MobileNetV2',
        tags: ['CNN', 'Mobile'],
        summary: 'A compact CNN designed for image classification with limited compute.',
        differs: 'Expands channels before depthwise convolution, then projects them into a narrow linear bottleneck.',
        facts: [
          ['Family', 'CNN, MobileNet'],
          ['Params', '3.5M'],
          ['Year', '2018'],
          ['Depth', '17 inverted residual blocks'],
          ['Read-out', 'Last conv block'],
        ],
        links: [
          { label: 'torchvision: MobileNetV2', href: 'https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html' },
        ],
      },
      {
        id: 'resnet101',
        title: 'ResNet-101',
        tags: ['CNN', 'Residual'],
        summary: 'A 101-layer residual network built from bottleneck blocks.',
        differs: 'Uses 1x1 convolutions around each 3x3 convolution to reduce computation, with more blocks than ResNet-18.',
        facts: [
          ['Family', 'CNN, residual'],
          ['Params', '44.5M'],
          ['Year', '2015'],
          ['Depth', '101 layers'],
          ['Read-out', 'Last convolution in layer4'],
        ],
        links: [
          { label: 'torchvision: ResNet-101', href: 'https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet101.html' },
        ],
      },
      {
        id: 'alexnet',
        title: 'AlexNet',
        tags: ['CNN', '2012'],
        summary: 'An eight-layer CNN that won ILSVRC 2012 and helped establish deep learning for image classification.',
        differs: 'Its first convolution is 11x11 with stride 4, so it discards spatial detail early and usually produces blockier maps.',
        facts: [
          ['Family', 'CNN'],
          ['Params', '61.1M'],
          ['Year', '2012'],
          ['Depth', '5 conv + 3 fully connected'],
          ['Read-out', 'Last conv block'],
        ],
        links: [
          { label: 'torchvision: AlexNet', href: 'https://docs.pytorch.org/vision/main/models/alexnet.html' },
        ],
      },
      {
        id: 'resnet18',
        title: 'ResNet-18',
        tags: ['CNN', 'Residual'],
        summary: 'An 18-layer CNN built from residual blocks with skip connections.',
        differs: 'Each residual block adds its input to its output. These shorter paths help gradients travel through the network and tend to produce cleaner gradient maps.',
        facts: [
          ['Family', 'CNN, residual'],
          ['Params', '11.7M'],
          ['Year', '2015'],
          ['Depth', '18 layers'],
          ['Read-out', 'layer4 (7x7)'],
        ],
        links: [
          { label: 'Paper (arXiv 1512.03385)', href: 'https://arxiv.org/abs/1512.03385' },
          { label: 'torchvision: ResNet', href: 'https://docs.pytorch.org/vision/main/models/resnet.html' },
        ],
      },
      {
        id: 'efficientnet_b0',
        title: 'EfficientNet-B0',
        tags: ['CNN', 'NAS'],
        summary: 'The baseline model for the EfficientNet family.',
        differs: 'Squeeze-and-excitation reweights complete channels using global context, so some of the evidence is not tied to a particular location.',
        facts: [
          ['Family', 'CNN, found by NAS'],
          ['Params', '5.3M'],
          ['Year', '2019'],
          ['Depth', 'Inverted residual blocks'],
          ['Read-out', 'Last conv block'],
        ],
        links: [
          { label: 'Paper (arXiv 1905.11946)', href: 'https://arxiv.org/abs/1905.11946' },
          { label: 'torchvision: EfficientNet', href: 'https://docs.pytorch.org/vision/main/models/efficientnet.html' },
        ],
      },
    ],
  },
  {
    kind: 'dataset',
    label: 'Datasets',
    entries: [
      {
        id: 'imagenet-pico',
        title: 'imagenet-pico',
        tags: ['Real photos'],
        summary: 'A small slice of the ImageNet-1k validation set with real photographs and original labels.',
        differs: 'This is the reference dataset: it shows how the methods behave on the type of data used to evaluate the models.',
        facts: [
          ['Images', 'A handful per class'],
          ['Classes', '1000 (ImageNet-1k label space)'],
          ['Source', 'ImageNet-1k validation split'],
          ['Preprocessing', 'Resize 256, centre-crop 224'],
        ],
        links: [
          { label: 'ImageNet', href: 'https://www.image-net.org/' },
        ],
      },
      {
        id: 'imagenet-pico-ai',
        title: 'imagenet-pico-ai',
        tags: ['Generated'],
        summary: 'Generated images covering the same classes as imagenet-pico.',
        differs: 'It is designed for comparison with imagenet-pico: the labels are the same, but the inputs are synthetic. Treat it as a probe, not as a benchmark comparable to published ImageNet results.',
        facts: [
          ['Images', 'Paired with imagenet-pico'],
          ['Classes', 'The same 1000'],
          ['Source', 'Generated'],
          ['Preprocessing', 'Resize 256, centre-crop 224'],
        ],
        links: [],
      },
    ],
  },
  {
    kind: 'method',
    label: 'Methods',
    entries: [
      {
        id: 'Saliency',
        title: 'Saliency',
        tags: ['Gradient'],
        summary: 'The absolute gradient of the class score with respect to each input pixel.',
        differs: 'It measures local sensitivity: which pixels would change the score most after a small change. That is a slope, not a direct measure of contribution.',
        facts: [
          ['Family', 'Gradient'],
          ['Needs', 'One backward pass'],
          ['Cost', 'Cheapest'],
          ['Resolution', 'Per pixel'],
          ['Limit', 'Speckled — neighbouring pixels can get very different gradients'],
        ],
        links: [
          { label: 'Paper (arXiv 1312.6034)', href: 'https://arxiv.org/abs/1312.6034' },
          { label: 'Captum: Saliency', href: `${CAPTUM}saliency.html` },
        ],
      },
      {
        id: 'InputXGradient',
        title: 'Input x Gradient',
        tags: ['Gradient'],
        summary: 'The gradient multiplied by the input value, giving a first-order estimate of each pixel\'s contribution.',
        differs: 'Multiplying by the input makes the result depend on both sensitivity and the evidence present in the image.',
        facts: [
          ['Family', 'Gradient'],
          ['Needs', 'One backward pass'],
          ['Cost', 'Cheap'],
          ['Resolution', 'Per pixel'],
          ['Limit', 'Still a point estimate, so it inherits gradient noise'],
        ],
        links: [
          { label: 'Captum: InputXGradient', href: `${CAPTUM}input_x_gradient.html` },
        ],
      },
      {
        id: 'IntegratedGradients',
        title: 'Integrated Gradients',
        tags: ['Gradient', 'Axiomatic'],
        summary: 'Averages gradients along a straight path from a baseline to the input.',
        differs: 'It satisfies completeness: the attributions add up to the difference between the model scores for the input and the baseline.',
        facts: [
          ['Family', 'Gradient, axiomatic'],
          ['Needs', 'Gradients along a path from a baseline'],
          ['Cost', 'Moderate'],
          ['Resolution', 'Per pixel'],
          ['Limit', 'A black baseline may miss evidence that is also dark'],
        ],
        links: [
          { label: 'Paper (arXiv 1703.01365)', href: 'https://arxiv.org/abs/1703.01365' },
          { label: 'Captum: IntegratedGradients', href: `${CAPTUM}integrated_gradients.html` },
        ],
      },
      {
        id: 'LayerIntegratedGradients',
        title: 'Layer Integrated Gradients',
        tags: ['Gradient', 'Layer'],
        summary: 'Integrated Gradients computed on an internal layer\'s activations rather than directly on the pixels.',
        differs: 'The attribution is computed on a feature map and then upsampled to the image, making it coarser and smoother than the input-space version.',
        facts: [
          ['Family', 'Gradient, layer'],
          ['Needs', 'Gradients along a path, at one layer'],
          ['Cost', 'Moderate'],
          ['Resolution', 'Feature map, upsampled'],
          ['Limit', 'The selected layer determines what the map can show'],
        ],
        links: [
          { label: 'Captum: Layer attribution', href: `${CAPTUM}layer.html` },
        ],
      },
      {
        id: 'DeepLift',
        title: 'DeepLIFT',
        tags: ['Gradient', 'Baseline'],
        summary: 'Propagates differences from a baseline activation backwards using per-layer rules.',
        differs: 'It can pass evidence through saturated neurons where a raw gradient is zero.',
        facts: [
          ['Family', 'Gradient, baseline'],
          ['Needs', 'One modified backward pass'],
          ['Cost', 'Cheap'],
          ['Resolution', 'Per pixel'],
          ['Limit', 'The reference activation affects the result'],
        ],
        links: [
          { label: 'Paper (arXiv 1704.02685)', href: 'https://arxiv.org/abs/1704.02685' },
          { label: 'Captum: DeepLift', href: `${CAPTUM}deep_lift.html` },
        ],
      },
      {
        id: 'DeepLiftShap',
        title: 'DeepLIFT SHAP',
        tags: ['Gradient', 'Shapley'],
        summary: 'DeepLIFT averaged over several baselines to approximate SHAP values.',
        differs: 'Using multiple baselines reduces dependence on a single reference, at the cost of one pass for each baseline.',
        facts: [
          ['Family', 'Gradient, Shapley'],
          ['Needs', 'One pass per baseline'],
          ['Cost', 'Moderate'],
          ['Resolution', 'Per pixel'],
          ['Limit', 'Runtime grows with the number of baselines'],
        ],
        links: [
          { label: 'Paper (arXiv 1705.07874)', href: 'https://arxiv.org/abs/1705.07874' },
          { label: 'Captum: DeepLiftShap', href: `${CAPTUM}deep_lift_shap.html` },
        ],
      },
      {
        id: 'GradientShap',
        title: 'Gradient SHAP',
        tags: ['Gradient', 'Shapley'],
        summary: 'Expected gradients: samples points between baselines and the input, then averages Input x Gradient at those points.',
        differs: 'Sampling several points and averaging their gradients generally produces smoother maps than plain Saliency.',
        facts: [
          ['Family', 'Gradient, Shapley'],
          ['Needs', 'Gradients at sampled noisy points'],
          ['Cost', 'Moderate'],
          ['Resolution', 'Per pixel'],
          ['Limit', 'It is an estimate, so repeated runs may differ'],
        ],
        links: [
          { label: 'Paper (arXiv 1705.07874)', href: 'https://arxiv.org/abs/1705.07874' },
          { label: 'Captum: GradientShap', href: `${CAPTUM}gradient_shap.html` },
        ],
      },
      {
        id: 'GuidedBackprop',
        title: 'Guided Backpropagation',
        tags: ['Gradient', 'Sharp'],
        summary: 'Backpropagation with negative gradients clipped at every ReLU, producing sharp maps with limited class sensitivity.',
        differs: 'Only positive gradients are propagated, so the maps emphasise edges. They often look similar across different target classes.',
        facts: [
          ['Family', 'Gradient, sharp'],
          ['Needs', 'One clipped backward pass'],
          ['Cost', 'Cheap'],
          ['Resolution', 'Per pixel'],
          ['Limit', 'It changes little between classes and may reflect the image more than the decision'],
        ],
        links: [
          { label: 'Paper (arXiv 1412.6806)', href: 'https://arxiv.org/abs/1412.6806' },
          { label: 'Captum: GuidedBackprop', href: `${CAPTUM}guided_backprop.html` },
        ],
      },
      {
        id: 'Deconvolution',
        title: 'Deconvolution',
        tags: ['Gradient', 'Sharp'],
        summary: 'A reconstruction method that uses the gradient sign at each ReLU instead of the forward activation.',
        differs: 'It is included as a historical reference. In sanity checks, its maps can remain similar even after the model weights are randomised.',
        facts: [
          ['Family', 'Gradient, sharp'],
          ['Needs', 'One modified backward pass'],
          ['Cost', 'Cheap'],
          ['Resolution', 'Per pixel'],
          ['Limit', 'Useful as a historical reference, but unreliable as an explanation'],
        ],
        links: [
          { label: 'Paper (arXiv 1311.2901)', href: 'https://arxiv.org/abs/1311.2901' },
          { label: 'Captum: Deconvolution', href: `${CAPTUM}deconvolution.html` },
        ],
      },
      {
        id: 'LayerGradCam',
        title: 'Grad-CAM',
        tags: ['Gradient', 'Layer'],
        summary: 'Weights a convolutional layer\'s feature maps by their average gradient to produce a coarse, class-specific map.',
        differs: 'It is useful for asking whether the model focused on the object or the background. It identifies a region, not individual pixels.',
        facts: [
          ['Family', 'Gradient, layer'],
          ['Needs', 'One backward pass at the last conv layer'],
          ['Cost', 'Cheap'],
          ['Resolution', 'Feature map (7x7 on ResNet-18)'],
          ['Limit', 'The map is upsampled to image size, so fine detail is not directly measured'],
        ],
        links: [
          { label: 'Paper (arXiv 1610.02391)', href: 'https://arxiv.org/abs/1610.02391' },
          { label: 'Captum: LayerGradCam', href: `${CAPTUM}layer.html` },
        ],
      },
      {
        id: 'GuidedGradCam',
        title: 'Guided Grad-CAM',
        tags: ['Gradient', 'Hybrid'],
        summary: 'The element-wise product of Guided Backprop and Grad-CAM, combining localisation with fine detail.',
        differs: 'The map is sharp and well localised, but its fine detail is not necessarily class-specific.',
        facts: [
          ['Family', 'Gradient, hybrid'],
          ['Needs', 'Grad-CAM times Guided Backprop'],
          ['Cost', 'Cheap'],
          ['Resolution', 'Per pixel'],
          ['Limit', 'Its fine detail inherits Guided Backprop\'s weak class sensitivity'],
        ],
        links: [
          { label: 'Paper (arXiv 1610.02391)', href: 'https://arxiv.org/abs/1610.02391' },
          { label: 'Captum: GuidedGradCam', href: `${CAPTUM}guided_grad_cam.html` },
        ],
      },
      {
        id: 'CB-RISE',
        title: 'CB-RISE',
        tags: ['Perturbation', 'Random masks', 'Blur'],
        summary: 'Extends RISE with blurred perturbations, automatic convergence detection and prediction-relative normalization.',
        differs: 'It replaces black masked regions with a blurred image and stops once the running variance of the map stabilises.',
        facts: [
          ['Family', 'Perturbation, Monte Carlo'],
          ['Needs', 'Up to 4096 masked forward passes, with early stopping'],
          ['Cost', 'Slow, but convergence-aware'],
          ['Resolution', '7x7 random masks, bilinearly upsampled'],
          ['Limit', 'Runtime and detail depend on the convergence and mask settings'],
        ],
        links: [
          { label: 'Paper (DOI)', href: 'https://doi.org/10.1007/978-3-031-70807-7_4' },
          { label: 'Original implementation', href: 'https://github.com/indirivacua/cbrise' },
        ],
      },
      {
        id: 'RISE',
        title: 'RISE',
        tags: ['Perturbation', 'Random masks'],
        summary: 'Averages many random masks, weighting each mask by the class score that remains visible.',
        differs: 'Unlike Occlusion, it uses many overlapping low-resolution masks instead of moving one fixed patch across the image.',
        facts: [
          ['Family', 'Perturbation, Monte Carlo'],
          ['Needs', '2048 masked forward passes, evaluated in batches'],
          ['Cost', 'Very slow'],
          ['Resolution', '7x7 random masks, bilinearly upsampled'],
          ['Limit', 'The colours show relative contrast; a warm colour does not by itself mean positive importance'],
        ],
        links: [
          { label: 'Paper (arXiv 1806.07421)', href: 'https://arxiv.org/abs/1806.07421' },
          { label: 'Original implementation', href: 'https://github.com/eclique/RISE' },
        ],
      },
      {
        id: 'Occlusion',
        title: 'Occlusion',
        tags: ['Perturbation'],
        summary: 'Slides a grey patch across the image and measures the resulting change in the class score.',
        differs: 'It uses only forward passes, so it provides a useful comparison point for gradient-based methods.',
        facts: [
          ['Family', 'Perturbation'],
          ['Needs', 'One forward pass per patch position'],
          ['Cost', 'Slow'],
          ['Resolution', 'The patch size'],
          ['Limit', 'The patch size determines the smallest region the map can resolve'],
        ],
        links: [
          { label: 'Paper (arXiv 1311.2901)', href: 'https://arxiv.org/abs/1311.2901' },
          { label: 'Captum: Occlusion', href: `${CAPTUM}occlusion.html` },
        ],
      },
      {
        id: 'Lime',
        title: 'LIME',
        tags: ['Perturbation', 'Superpixels'],
        summary: 'Fits a sparse linear model to perturbed versions of the image, using superpixel segments as features.',
        differs: 'The explanation comes from the local surrogate model: its coefficients estimate the effect of switching segments on and off.',
        facts: [
          ['Family', 'Perturbation, superpixels'],
          ['Needs', 'A few hundred forward passes'],
          ['Cost', 'Slow'],
          ['Resolution', 'One value per segment'],
          ['Limit', 'The segmentation strongly affects the result'],
        ],
        links: [
          { label: 'Paper (arXiv 1602.04938)', href: 'https://arxiv.org/abs/1602.04938' },
          { label: 'Captum: Lime', href: `${CAPTUM}lime.html` },
        ],
      },
      {
        id: 'KernelShap',
        title: 'Kernel SHAP',
        tags: ['Perturbation', 'Shapley'],
        summary: 'A LIME-style surrogate with the SHAP kernel, so its segment weights approximate Shapley values.',
        differs: 'It uses the same perturb-and-regress process as LIME, but changes the sample weighting to match the SHAP formulation.',
        facts: [
          ['Family', 'Perturbation, Shapley'],
          ['Needs', 'A few hundred forward passes'],
          ['Cost', 'Slow'],
          ['Resolution', 'One value per segment'],
          ['Limit', 'It depends on the segmentation just as LIME does'],
        ],
        links: [
          { label: 'Paper (arXiv 1705.07874)', href: 'https://arxiv.org/abs/1705.07874' },
          { label: 'Captum: KernelShap', href: `${CAPTUM}kernel_shap.html` },
        ],
      },
      {
        id: 'ShapleyValueSampling',
        title: 'Shapley Value Sampling',
        tags: ['Perturbation', 'Shapley'],
        summary: 'Estimates Shapley values by sampling segment orderings and measuring each segment\'s marginal contribution.',
        differs: 'It estimates the contributions directly, without a surrogate model, which makes it the most expensive Shapley method here.',
        facts: [
          ['Family', 'Perturbation, Shapley'],
          ['Needs', 'One pass per segment, per ordering'],
          ['Cost', 'Slowest'],
          ['Resolution', 'One value per segment'],
          ['Limit', 'Runtime grows with the number of segments'],
        ],
        links: [
          { label: 'Paper (arXiv 1705.07874)', href: 'https://arxiv.org/abs/1705.07874' },
          { label: 'Captum: ShapleyValueSampling', href: `${CAPTUM}shapley_value_sampling.html` },
        ],
      },
      {
        id: 'FeatureAblation',
        title: 'Feature Ablation',
        tags: ['Perturbation'],
        summary: 'Replaces one feature group at a time with a baseline and measures the change in the output.',
        differs: 'Like Occlusion, but with explicitly defined groups instead of a sliding grid. The groups can therefore follow image segments.',
        facts: [
          ['Family', 'Perturbation'],
          ['Needs', 'One forward pass per group'],
          ['Cost', 'Slow'],
          ['Resolution', 'One value per group'],
          ['Limit', 'A constant baseline can create inputs outside the model\'s training distribution'],
        ],
        links: [
          { label: 'Captum: FeatureAblation', href: `${CAPTUM}feature_ablation.html` },
        ],
      },
      {
        id: 'FeaturePermutation',
        title: 'Feature Permutation',
        tags: ['Perturbation'],
        summary: 'Shuffles a feature group across the batch and measures the change in the output.',
        differs: 'The replacement values come from other samples in the batch, preserving the feature\'s observed distribution better than a constant baseline.',
        facts: [
          ['Family', 'Perturbation'],
          ['Needs', 'One pass per group, over a batch'],
          ['Cost', 'Slow'],
          ['Resolution', 'One value per group'],
          ['Limit', 'It needs a batch from which to draw replacement features'],
        ],
        links: [
          { label: 'Captum: FeaturePermutation', href: `${CAPTUM}feature_permutation.html` },
        ],
      },
      {
        id: '__segmentation__',
        title: 'SLIC / Quickshift / KMeans',
        tags: ['Segmentation'],
        summary: 'The suffix on a perturbation method identifies the algorithm used to divide the image into segments.',
        differs: 'SLIC creates compact grid-based regions, Quickshift follows colour-density modes and produces irregular regions, and KMeans clusters pixels without a spatial constraint.',
        facts: [
          ['Family', 'Segmentation, not attribution'],
          ['Needs', 'One pass, before the method runs'],
          ['Cost', 'Negligible'],
          ['Resolution', 'It sets the segment size'],
          ['Limit', 'Changing the segmentation can change the map even when the method stays the same'],
        ],
        links: [
          { label: 'scikit-image: segmentation', href: 'https://scikit-image.org/docs/stable/api/skimage.segmentation.html' },
        ],
      },
    ],
  },
  {
    kind: 'metric',
    label: 'Metrics',
    entries: [
      {
        id: 'lif',
        title: 'LIF — Least Important First',
        tags: ['Faithfulness'],
        summary: 'Deletes pixels in ascending order of attribution and reports the area under the class-score curve.',
        differs: 'Removing pixels marked as unimportant should initially have little effect, so a faithful map should keep the score high over most of the curve.',
        facts: [
          ['Measures', 'Faithfulness, by deletion'],
          ['Better', 'Higher'],
          ['Removes', 'Pixels, least-attributed first'],
          ['Limit', 'A single deletion order; it never tests removing the top-ranked pixels first'],
        ],
        links: [],
      },
      {
        id: 'morph',
        title: 'Morph — morphological faithfulness',
        tags: ['Faithfulness'],
        summary: 'Erodes the most-attributed region step by step and reports the area under the class-score curve.',
        differs: 'Shrinking one contiguous region produces more coherent perturbations than deleting scattered pixels, but gives a coarser measurement.',
        facts: [
          ['Measures', 'Faithfulness, by preservation'],
          ['Better', 'Higher'],
          ['Removes', 'Everything outside the region, then its border inward'],
          ['Limit', 'It is coarser than the pixel-wise curves'],
        ],
        links: [],
      },
      {
        id: 'segment',
        title: 'Segment — segment-wise deletion',
        tags: ['Faithfulness'],
        summary: 'Deletes segments in ascending order of mean attribution and reports the area under the class-score curve.',
        differs: 'Using superpixels as the unit of removal makes the metric applicable to both pixel-level and segment-level methods.',
        facts: [
          ['Measures', 'Faithfulness, by deletion'],
          ['Better', 'Higher'],
          ['Removes', 'Superpixels, least-attributed first'],
          ['Limit', 'The result depends on the segmentation used'],
        ],
        links: [],
      },
      {
        id: 'fidelity_superpixel',
        title: 'Fidelity — superpixel removal',
        tags: ['Faithfulness'],
        summary: 'LIME and KernelSHAP fidelity when one uniformly sampled attribution superpixel is replaced by zero per perturbation.',
        differs: 'Uses the same segmentation as the explanation and counts each feature coefficient once. Reported alongside standard fidelity; different removal shapes and areas make the scores separate experiments.',
        facts: [
          ['Methods', 'LIME and KernelSHAP, SLIC or KMeans'],
          ['Better', 'Higher'],
          ['Range', '(-∞, 1]'],
          ['Sampling', 'One whole feature per draw, with replacement'],
          ['Limit', 'Undefined when the model never responds to the sampled removals'],
        ],
        links: [],
      },
      {
        id: 'fidelity',
        title: 'Fidelity — relative error',
        tags: ['Faithfulness'],
        summary: 'The relative reduction in prediction error over a zero attribution under random perturbations.',
        differs: 'It normalizes infidelity against an explanation that always predicts zero change: 1 is exact, 0 matches that baseline, and negative values are worse.',
        facts: [
          ['Measures', 'Relative explanation skill'],
          ['Better', 'Higher'],
          ['Range', '(-∞, 1]'],
          ['Removes', 'Nothing — it perturbs at random'],
          ['Limit', 'Undefined when the model never responds to the sampled perturbations'],
        ],
        links: [
          { label: 'Paper (arXiv 1901.09392)', href: 'https://arxiv.org/abs/1901.09392' },
        ],
      },
    ],
  },
];

const INDEX = new Map(
  WIKI_SECTIONS.flatMap((s) => s.entries.map((e) => [`${s.kind}:${e.id.toLowerCase()}`, e]))
);

const KIND_OF = new Map(
  WIKI_SECTIONS.flatMap((s) => s.entries.map((e) => [e, s.kind]))
);

// "Lime (SLIC)" -> "Lime". Segmentation variants share one entry.
function baseMethodId(id) {
  return String(id).replace(/\s*\([^)]*\)\s*$/, '').trim();
}

export function lookupWiki(kind, id) {
  if (!id) return null;
  const direct = INDEX.get(`${kind}:${String(id).toLowerCase()}`);
  if (direct) return direct;
  if (kind !== 'method') return null;
  return INDEX.get(`method:${baseMethodId(id).toLowerCase()}`) ?? null;
}

export function wikiKindOf(entry) {
  return KIND_OF.get(entry) ?? null;
}
