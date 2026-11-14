import Link from "next/link";
import { BRAND } from "@/content/brand";
import { PurposeSwitch } from "@/components/welcome/PurposeSwitch";
import { ABOUT } from "@/content/copy";
import { exploreHref, pilotFrontDoor } from "@/pilot/frontDoor";
import { AboutKinds, Publication, Providers } from "@/nx/pages/about";
import { InfoPage, InfoSection, infoStyles as s } from "@/nx/pages/InfoPage";
import { registryTrust } from "@/nx/pages/registry.server";

export const dynamic = "force-dynamic";
export const metadata = { title: "About" };

const CONTENTS = [
  { id: "purpose-h", label: "Set up for" },
  { id: "kinds-h", label: "Four kinds of information" },
  { id: "cannot-h", label: "What it cannot tell you" },
  { id: "research-h", label: "The research" },
  { id: "providers-h", label: "Data and methods" },
  { id: "review-h", label: "Review status" },
];

/**
 * What Neritic is, the kinds of information it shows and where, what it cannot tell you, the research it comes from
 * (and what that research does not cover), and its data providers. v15.1: a page of the shell (src/nx/pages); the
 * words are src/content/copy.ts ABOUT and RESEARCH, unchanged.
 */
export default async function AboutPage() {
  return (
    <InfoPage
      id="about"
      testId="about-page"
      kicker={`About Neritic · ${BRAND.tagline}`}
      title={ABOUT.title}
      lede={ABOUT.lede}
      trust={await registryTrust()}
      contents={CONTENTS}
      head={
        <>
          <p>
            <Link href={exploreHref(pilotFrontDoor())} className={s.link} data-testid="about-tour">
              Open Explore, the map →
            </Link>
          </p>
          <p className={s.small} data-testid="about-independence">
            {BRAND.independence} {BRAND.formerly}
          </p>
        </>
      }
    >
      <InfoSection id="purpose-h" title="What Neritic is set up for on this device" data-testid="about-purpose">
        <PurposeSwitch showReset />
        <p className={`${s.small} ${s.after}`}>
          It changes only where you start and what comes first. Every page, value, caveat and official notice is the same for everyone. Nothing is sent anywhere.
        </p>
      </InfoSection>

      <InfoSection id="kinds-h" title="Four kinds of information, never mixed" intro={ABOUT.kindsIntro}>
        <AboutKinds />
      </InfoSection>

      <InfoSection id="cannot-h" title="What Neritic cannot tell you" data-testid="about-cannot">
        <div className={s.prose}>
          <ul>
            {ABOUT.cannot.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </div>
      </InfoSection>

      <InfoSection id="research-h" title="The research behind it" data-testid="research">
        <Publication />
        <div className={s.prose}>
          <p>{ABOUT.research.what}</p>
          <p>{ABOUT.research.result}</p>
          <p className={s.note} data-testid="research-relation">
            {ABOUT.research.relation}
          </p>
        </div>
      </InfoSection>

      <InfoSection id="providers-h" title="Data and methods" intro={ABOUT.providersIntro}>
        <Providers />
        <p className={s.links}>
          <Link href="/sources" className={s.link} data-testid="about-sources">
            Data status and licences →
          </Link>
          <a href={ABOUT.repo} target="_blank" rel="noreferrer" className={s.link}>
            Code, methods and documentation ↗
          </a>
        </p>
      </InfoSection>

      <InfoSection id="review-h" title="Review status">
        <div className={s.prose}>
          <ul data-testid="about-review">
            {ABOUT.review.map((t) => (
              <li key={t}>{t}</li>
            ))}
          </ul>
        </div>
      </InfoSection>
    </InfoPage>
  );
}
