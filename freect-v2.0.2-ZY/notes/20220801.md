# Staying focused on Goals

Been working on FreeCT a fair bit lately and have finally made it to the core WFBP code; the bit that actually does the reconstruction.  Ah how the memories are flooding back.  Lately, as I'm programming, if I feel like I'm getting stuck or the code isn't flowing as freely as I'd expect it to, it is often reflective of some bigger, unanswered question that I've missed, or an incorrect design choice I made that should likely be revisited.  Today's case specifically was brough about by the rebinning code for WFBP.

**Tradeoffs**

The core of the issue is the tradeoff between efficiency and speed and, well, everything else (readability, maintainability, memory management, etc.)  It's significantly more computationally efficient to rebin and filter as one big step.  In fact, in a perfect world with infinite GPU memory, we'd do one giant copy of our raw projection data onto the GPU, and then one final copy back of our reconstructed slices.  The only problem there though is that you need a GPU with a LOT of memory for clinical CT.  Considering that we live in the real world, and I want FreeCT to be useable on other devices than just a top-of-the-line GPU, I need to think a little more critically about my approach.

The rebinning code reminded me of these tradeoffs and decisions I have to be making as part of this rewrite, and brought me back to focusing on design-driven engineering.  This rewrite is primarily focused on improving FreeCT **usability**, with secondary goal(s) of maintainability and reusability.  Only as a tertiary consideration am I focusing on speed inasmuch as FreeCT needs to be suitable for lots of clinical quality reconstructions in reasonable amount of time.

To summarize the goals of porting FreeCT to Go (as much for my own benefit as anyone else's):

1. Usability
    - Binaries for Windows and Linux (maybe one day Mac, but lack of Nvidia support limits the urgency here)
    - Docker container
    - Plug-and-play support for TCIA LDCT dataset (more broadly, the Battelle format for raw data)
    - Support for GPUs with 4GB of memory
2. Maintainability and Reusability
    - Code should be clear and easy-to-read
    - Components (raw data readers/writers, maybe things like rebinning) should be interposable accross different algorithms and tools
3. Speed
    - FreeCT should be easy to automate (this is really where FreeCT is a speed improvement over having to go back to the scanner)
    - Fast... enough (let's set an SLR of 5 minutes for a "clinical quality" reconstruction)
   
This clarifies things at least for now, that rebinning, filtering, and backprojection should be standalone steps with sane, self-evident inputs and outputs, even if it means a slower overall reconstruction.  I'm not 100% clear exactly how this will shake out as it does tend to result in a fair bit of extra computation (which will be hard for me to totally get over,) but I do think it will make for more a more reasonable end-result.  It'll also make it more testable, which could dramatically improve long-term maintainability.

**Managing Memory Use**

Somewhat tangentially related to usability, I'm make two choices with respect to memory that I think it worth mentioning.  The first is that I will assume we are not limited by system memory.  RAM is cheap enough these days that I don't think it's a worthwhile use of my time to carefully optimize how much is in use at any point.  Historically, this has meant that FreeCT might use ~8GB at its peak during reconstruction; on the other hand, I don't fully know what that will look like though in the world of Go when I'm not explicitly managing memory.  Given that many machines these days have at least 16GB, I don't expect this to be much of an issue, however if it proves to be for you and your work, please reach out.  I'm not opposed to spending some effort on this, I just don't think it's an especially pressing issue at the moment.

The second choice is to focus FreeCT primarily on supporting GPUs with around 4GB of memory.  Although there are examples of cards out there with less memory, I think this is the sweet spot balancing broad availity (cards in this memory range can be had for $100-200 brand new, and much cheaper on ebay), and *enough* memory to still get fast, meaningful work done.  FreeCT will include a parameter to use *more* memory when available, however 4GB will be the default.  I'll also be sure to leave some headroom though if you're running on your main display card.

On that note, I should get to writing some code.

** John - 2022-08-01 **