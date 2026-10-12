# Independent scientific review: request draft (not sent) and reviewer checklist

**Draft only. Nothing has been sent.** The owner chooses the recipient. Suitable reviewers have worked with C-HARM, CalHABMAP shore-station data or domoic acid management in California, and have no role in building CoastWatch. Before sending, replace the bracketed fields and check that the preview link works for the recipient: it needs a Vercel login today, so either invite their account or arrange a short screen-share.

---

**Subject:** Request for an independent scientific review of CoastWatch, a California HAB information site

Dear [Dr. Name],

I am [Name], [role/affiliation]. I am preparing a non-commercial, open-source website, CoastWatch, that brings together public information on harmful algal blooms along the California coast. Before making it public, I would value an independent review from someone with your expertise in [C-HARM / domoic acid monitoring / California HAB management].

CoastWatch shows four kinds of information, each labelled:
- NOAA's C-HARM probabilities;
- CalHABMAP shore-station measurements via SCCOOS ERDDAP, along with satellite chlorophyll;
- CDFW and CDPH notices, transcribed and linked to their sources;
- historical fisheries landings for HAB-sensitive species.

It does not produce its own forecasts or toxin estimates. It does not decide whether an area is open or seafood is safe.

I would be grateful for your view on four questions:
1. **C-HARM:** are the probabilities, thresholds, forecast days and nearshore gaps described correctly, and is anything implied that the model does not support?
2. **Station observations:** are domoic acid fractions, Pseudo-nitzschia counts, reported zeros and reporting delays presented accurately, and is anything over-interpreted?
3. **Fisheries species tiers:** is the grouping of species by HAB sensitivity reasonable, and is its basis stated fairly?
4. **Scientific wording:** is any statement misleading, overstated or likely to be misread by the public?

The review would take about [1–2] hours: a walkthrough of four pages and a short checklist (attached). Written comments or a 30-minute call would both be very welcome. I will record your findings and the changes made. I will not describe CoastWatch as reviewed or endorsed by you, or name you, without your explicit permission.

The site is not public yet. I can give you private access to the preview [link/arrangement]. The code and documentation are at https://github.com/yashnil/habs-forecast. The related research publication is "Physics-Guided Neural Forecasts of Nearshore Harmful Algal Blooms in the California Current System" (https://doi.org/10.33422/ccgconf.v2i2.1619). It concerns chlorophyll forecasting; CoastWatch does not run those models.

Thank you for considering this. I understand if your time does not allow it, and would welcome a suggestion of someone else to ask.

Kind regards,
[Name]
[Contact]

---

## Reviewer checklist

For each item, please mark **OK**, **Change** (with a suggestion) or **Not my area**. Page names refer to the site's navigation.

### A. C-HARM (Ocean Map; Bloom Intelligence "Forecast near the station")
1. The three layers are described as probabilities that a threshold is exceeded:
   - Pseudo-nitzschia above 10,000 cells/L;
   - particulate domoic acid above 500 ng/L;
   - cellular domoic acid above 10 pg/cell.

   Are these thresholds and their wording correct for C-HARM v3.1?
2. Nowcast and 1–3-day forecasts are labelled with the model's own valid dates. Is it right to present them as "model", never as observed conditions?
3. Cells without a value near shore are explained as "a nearshore cell the model does not cover", with the note that C-HARM lacks domoic acid probabilities for many cells within about 3–6 km of shore. Is that accurate?
4. At a shore station the site shows the C-HARM value near the station beside the measurements, and deliberately does not compare them numerically. Is that the right choice?
5. Is there anything a public user is likely to infer from the map colours that the model does not support (for example, toxin in seafood, or safety)?

### B. Station observations (Bloom Intelligence)
1. Particulate, dissolved and total domoic acid are kept separate (ng/mL of seawater) and never combined. Is the explanation of each fraction correct?
2. Pseudo-nitzschia counts are shown by size group (seriata, delicatissima) and never summed. Is that right?
3. A reported 0 is shown as "reported 0 (not quantified)", because the dataset publishes no detection limit. Is this accurate, and is the wording clear?
4. Values above a plausibility ceiling are kept and flagged, not removed. Are the ceilings sensible (for example, particulate domoic acid above 500 ng/mL)?
5. The site says a measurement describes the water at that pier on that day, is reported after a laboratory delay, and does not describe nearby waters or seafood. Is anything missing?
6. Is the "since 2014" history presented without implying trends the sampling cannot support?

### C. Fisheries species tiers (Fisheries)
1. Tier 1, fisheries with commercial closures or take restrictions for marine toxins (current or past): Dungeness crab, rock crabs, northern anchovy.
2. Tier 2, species under consumption advisories, sport-harvest quarantines or monitoring: spiny lobster, Pacific sardine, bivalves.
3. Are these assignments and their stated bases reasonable? Is any HAB-sensitive species missing (for example, razor clams, which are recreational only)? Is anything misleadingly included?
4. The page shows historical landings, inflation-adjusted, and says it does not show losses or forecasts. Is the 2015–16 domoic acid event described accurately?

### D. Scientific wording (all pages, especially About)
1. Chlorophyll is never presented as a toxin, and satellite chlorophyll is called a bloom proxy. Is that consistently true?
2. Any sentence that overstates certainty, causation or coverage. Please quote it.
3. The About page's description of the publication: does it avoid implying that the paper validates the application?
4. Anything you would want a member of the public to understand that the site does not say.

### E. Overall
- Would you be comfortable with this site being public, as is, after the changes you list? (Your answer is recorded privately; it is not an endorsement.)
