# Failure analysis

I read through every failure by hand: 15 from the BM25 run and 1 from the embeddings + cross-encoder run. 

## 1. The customer uses different words than the article (BM25 never finds it)

BM25 only matches words. If the customer says it differently, the right article doesn't show up.

- **Q15** "Can I send back something I bought by mistake?" The assistant refused. The right article (KB-12) is about returns, and "send back" shares no word with it.
- **Q27** "I can't remember how to get into my profile anymore." Refused. The password reset article (KB-20) never uses "get into" or "profile".
- **Q05** "My package has been stuck on the same status for a week." Refused. The article talks about days, not "a week", and describes the problem in other words (to check).

Embeddings fix all three, because they match meaning instead of words.

## 2. The right article was found, then thrown away

Here the search worked. The coverage check is what failed: it wants at least half of the question's words to appear in the article.

- **Q17** "Can you deliver to Switzerland?" KB-11 was ranked first, but the article never says "Switzerland" (it says it does not ship outside the EU). That rare word counts the most, so coverage drops under the threshold and the assistant refuses.
- **Q22** "I received a broken item. How fast must I report it?" KB-16 was ranked first. The article probably says "damaged" and "48 hours" (to check), so coverage is too low and it refuses.
- **Q10** "How fast is Express delivery and what is the order cut-off time?" The question gets split on "and". Neither half passes the check, so nothing is left to answer with.
- **Q02** "Which payment methods can I use?" KB-02 was ranked first but not used. The answer came from the refund article (KB-15) because it contains "payment method", and it left out PayPal and SEPA.

The coverage check measures shared words, not whether the article is actually right. That is the weak point.

## 3. Right article, wrong sentence

The assistant cites the correct article but picks the sentence that looks most like the question, not the one with the fact.

- **Q16** Countries outside France: it picked "We do not ship outside the European Union" and skipped the list of countries.
- **Q33** E-bike battery with Express: it picked a sentence about shipping inside France only and missed that Express is not available.
- **Q20** Start a return: it gave the first step and dropped the label-printing step.
- **Q25** Cancel an order: it gave the Received/Preparing rule and left out the "packed" detail.
- **Q29** Delete an account: it gave the click path and left out the 30-day delay.
- **Q31** Unsubscribe: it gave the link and left out the 48-hour delay.

Sentences with delays, exceptions or next steps share few words with the question, so they get cut by the 3-sentence limit and the `rho` ratio.

## 4. A different article answers, and it sounds right

This is the worst case, because the user gets a confident answer to the wrong problem.

- **Q04** "How many days do I have to return an item?" (BM25) It answered with the "how to start a return" steps (KB-14), which contain "return" and "item" many times. The articles with the 30 and 14 day figures ranked lower.
- **Q08** "How long do refunds take?" (BM25) It answered from the lost-parcel article: "up to 10 working days". That is a real delay, just not the refund delay.
- **Q27** (embeddings + cross-encoder, the one remaining failure) The right article, KB-20, was retrieved second. The cross-encoder put the account deletion article first, because "my profile" looks a lot like "my account". The answer was "Go to Settings, then Privacy, then Delete my account." Someone who forgot their password is told how to delete their account, with no sign of doubt. Its score (0.53) was barely above the threshold (0.50), but nothing in the answer shows that.
