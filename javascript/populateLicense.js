function fetchLicense(id, url) {
    fetch(url).then((response) => {
        response.text().then(
            licenseText => document.getElementById(id).textContent = licenseText
        );
    })
}

function populateLicense() {
    const pairs = [
        ["sd1", "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/LICENSE"],
        ["sdxl", "https://raw.githubusercontent.com/Stability-AI/generative-models/main/LICENSE-CODE"],
        ["flux", "https://raw.githubusercontent.com/black-forest-labs/flux/main/LICENSE"],
        ["comfy", "https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/LICENSE"],
        ["chain", "https://raw.githubusercontent.com/chaiNNer-org/chaiNNer/main/LICENSE"],
        ["tfm", "https://raw.githubusercontent.com/huggingface/transformers/main/LICENSE"],
        ["dot", "https://raw.githubusercontent.com/huggingface/diffusers/main/LICENSE"],
        ["invoke", "https://raw.githubusercontent.com/invoke-ai/InvokeAI/main/LICENSE"],
        ["mem", "https://raw.githubusercontent.com/AminRezaei0x443/memory-efficient-attention/main/LICENSE"],
        ["ctfm", "https://raw.githubusercontent.com/explosion/curated-transformers/main/LICENSE"],
        ["taesd", "https://raw.githubusercontent.com/madebyollin/taesd/main/LICENSE"],
        ["ldsr", "https://raw.githubusercontent.com/hafiidz/latent-diffusion/main/LICENSE"],
        ["clip", "https://raw.githubusercontent.com/pharmapsychotic/clip-interrogator/main/LICENSE"],
    ];

    for (const [id, url] of pairs)
        fetchLicense(`${id}-license-content`, url);
}
