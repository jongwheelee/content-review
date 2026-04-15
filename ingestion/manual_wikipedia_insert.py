"""Insert manually provided Wikipedia articles into the database.

This script inserts Wikipedia articles that were skipped due to disambiguation pages,
when the user provides the content manually.
"""

import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sentence_transformers import SentenceTransformer

from database.connection import db
from database.models import ResearchPaper, Embedding, PaperType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Manually provided Wikipedia articles
MANUAL_ARTICLES = {
    "Bond": """In finance, a bond is a type of security under which the issuer (debtor) owes the holder (creditor) a debt, and is obliged – depending on the terms – to provide cash flow to the creditor; which usually consists of repaying the principal (the amount borrowed) of the bond at the maturity date, as well as interest (called the coupon) over a specified amount of time. The timing and the amount of cash flow provided varies, depending on the economic value that is emphasized upon, thus giving rise to different types of bonds. Bonds and stocks are both securities, but the major difference between the two is that (capital) stockholders have an equity stake in a company (i.e. they are owners), whereas bondholders have a creditor stake in a company (i.e. they are lenders). As creditors, bondholders have priority over stockholders. This means they will be repaid in advance of stockholders, but will rank behind secured creditors, in the event of bankruptcy. Another difference is that bonds usually have a defined term, or maturity, after which the bond is redeemed, whereas stocks typically remain outstanding indefinitely. An exception is an irredeemable bond, which is a perpetuity, that is, a bond with no maturity. Certificates of deposit (CDs) or short-term commercial paper are classified as money market instruments and not bonds: the main difference is the length of the term of the instrument. Bonds are often identified by their international securities identification number, or ISIN, which is a 12-digit alphanumeric code that uniquely identifies debt securities.""",

    "Equity": """In finance, equity is an ownership interest in property that may be subject to debts or other liabilities. Equity is measured for accounting purposes by subtracting liabilities from the value of the assets owned. For example, if someone owns a car worth $24,000 and owes $10,000 on the loan used to buy the car, the difference of $14,000 is equity. Equity can apply to a single asset, such as a car or house, or to an entire business. A business that needs to start up or expand its operations can sell its equity in order to raise cash that does not have to be repaid on a set schedule. When liabilities attached to an asset exceed its value, the difference is called a deficit and the asset is informally said to be "underwater" or "upside-down". In government finance or other non-profit settings, equity is known as "net position" or "net assets". The term "equity" describes this type of ownership in English because it was regulated through the system of equity law that developed in England during the Late Middle Ages to meet the growing demands of commercial activity.""",

    "Alpha": """Alpha is a measure of the active return on an investment, the performance of that investment compared with a suitable market index. An alpha of 1% means the investment's return on investment over a selected period of time was 1% better than the market during that same period; a negative alpha means the investment underperformed the market. Alpha, along with beta, is one of two key coefficients in the capital asset pricing model used in modern portfolio theory and is closely related to other important quantities such as standard deviation, R-squared and the Sharpe ratio. In modern financial markets, where index funds are widely available for purchase, alpha is commonly used to judge the performance of mutual funds and similar investments. As these funds include various fees normally expressed in percent terms, the fund has to maintain an alpha greater than its fees in order to provide positive gains compared with an index fund. Historically, the vast majority of traditional funds have had negative alphas, which has led to a flight of capital to index funds and non-traditional hedge funds.""",

    "RSI": """The relative strength index (RSI) is a technical indicator used in the analysis of financial markets. It is intended to chart the current and historical strength or weakness of a stock or market based on the closing prices of a recent trading period. The indicator should not be confused with relative strength. The RSI is classified as a momentum oscillator, measuring the velocity and magnitude of price movements. Momentum is the rate of the rise or fall in price. The relative strength RS is given as the ratio of higher closes to lower closes. The RSI is most typically used on a 14-day timeframe, measured on a scale from 0 to 100, with high and low levels marked at 70 and 30, respectively. Short or longer timeframes are used for alternately shorter or longer outlooks. High and low levels—80 and 20, or 90 and 10—occur less frequently but indicate stronger momentum. The relative strength index was developed by J. Welles Wilder and published in a 1978 book, New Concepts in Technical Trading Systems.""",

    "Leverage": """In finance, leverage, also known as gearing, is any technique involving borrowing funds to buy an investment. Financial leverage is named after a lever in physics, which amplifies a small input force into a greater output force. Financial leverage uses borrowed money to augment the available capital, thus increasing the funds available for (perhaps risky) investment. If successful this may generate large amounts of profit. However, if unsuccessful, there is a risk of not being able to pay back the borrowed money. Normally, a lender will set a limit on how much risk it is prepared to take, and will set a limit on how much leverage it will permit. It would often require the acquired asset to be provided as collateral security for the loan. While leverage magnifies profits when the returns from the asset more than offset the costs of borrowing, leverage may also magnify losses. A corporation that borrows too much money might face bankruptcy or default during a business downturn, while a less-leveraged corporation might survive.""",

    "Derivatives": """In finance, a derivative is a contract between a buyer and a seller. The derivative can take various forms, depending on the transaction, but every derivative has the following four elements: an item (the "underlier") that can or must be bought or sold, a future act which must occur (such as a sale or purchase of the underlier), a price at which the future transaction must take place, and a future date by which the act (such as a purchase or sale) must take place. A derivative's value depends on the performance of the underlier, which can be a commodity (for example, corn or oil), a financial instrument (e.g. a stock or a bond), a price index, a currency, or an interest rate. Derivatives can be used to insure against price movements (hedging), increase exposure to price movements for speculation, or get access to otherwise hard-to-trade assets or markets. Some of the more common derivatives include forwards, futures, options, swaps, and variations of these such as synthetic collateralized debt obligations and credit default swaps.""",

    "Options": """In finance, an option is a contract which conveys to its owner, the holder, the right, but not the obligation, to buy or sell a specific quantity of an underlying asset or instrument at a specified strike price on or before a specified date, depending on the style of the option. Options are typically acquired by purchase, as a form of compensation, or as part of a complex financial transaction. Thus, they are also a form of asset (or contingent liability) and have a valuation that may depend on a complex relationship between underlying asset price, time until expiration, market volatility, the risk-free rate of interest, and the strike price of the option. An option that conveys to the holder the right to buy at a specified price is referred to as a call, while one that conveys the right to sell at a specified price is known as a put. The issuer may grant an option to a buyer as part of another transaction, or the buyer may pay a premium to the issuer for the option.""",

    "Futures": """In finance, a futures contract (sometimes called futures) is a standardized legal contract to buy or sell something at a predetermined price for delivery at a specified time in the future, between parties not yet known to each other. The item transacted is usually a commodity or financial instrument. The predetermined price of the contract is known as the forward price or delivery price. The specified time in the future when delivery and payment occur is known as the delivery date. Because it derives its value from the value of the underlying asset, a futures contract is a derivative. Futures contracts are widely used for hedging price risk and for speculative trading in commodities, currencies, and financial instruments. Futures contracts are traded at futures exchanges, which act as a marketplace between buyers and sellers. The buyer of a contractual right is said to be the long position holder and the selling party is said to be the short position holder.""",
}


async def insert_manual_articles():
    """Insert manually provided Wikipedia articles and generate embeddings."""
    logger.info("Starting manual Wikipedia article insertion")

    db.initialize()
    model = SentenceTransformer("all-MiniLM-L6-v2")

    total_inserted = 0
    total_embedded = 0
    errors = []

    async with db.async_session() as session:
        for topic, content in MANUAL_ARTICLES.items():
            external_id = f"wiki_{topic.lower().replace(' ', '_')}"

            # Check if already exists
            existing = await session.execute(
                select(ResearchPaper.id).where(ResearchPaper.external_id == external_id)
            )
            existing_id = existing.scalar_one_or_none()

            if existing_id:
                logger.info(f"  Already exists: {topic}")
                # Update existing
                paper = await session.get(ResearchPaper, existing_id)
                if paper:
                    paper.abstract = content
                    paper.updated_at = datetime.now()
                    paper.processed = False  # Re-process for embedding
                logger.info(f"  Updated: {topic}")
            else:
                # Insert new
                paper = ResearchPaper(
                    external_id=external_id,
                    title=topic,
                    authors=None,
                    abstract=content,
                    published_date=None,
                    url=f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                    keywords=[topic],
                    paper_type=PaperType.WIKIPEDIA,
                    processed=False,
                )
                session.add(paper)
                total_inserted += 1
                logger.info(f"  Inserted: {topic}")

            await session.commit()

        # Generate embeddings for all manual articles
        logger.info("Generating embeddings for manual articles...")
        for topic, content in MANUAL_ARTICLES.items():
            external_id = f"wiki_{topic.lower().replace(' ', '_')}"

            result = await session.execute(
                select(ResearchPaper.id).where(ResearchPaper.external_id == external_id)
            )
            paper_id = result.scalar_one()

            # Check if embedding already exists
            existing_embedding = await session.execute(
                select(Embedding.id).where(
                    Embedding.source_type == "research_paper",
                    Embedding.source_id == paper_id,
                )
            )
            if existing_embedding.scalar_one_or_none():
                logger.info(f"  Embedding exists: {topic}")
                continue

            try:
                # Generate embedding
                embedding_vector = model.encode(content)

                # Insert embedding
                embedding_record = {
                    "source_type": "research_paper",
                    "source_id": paper_id,
                    "content_text": content,
                    "embedding": embedding_vector.tolist(),
                    "meta": {"paper_type": "wikipedia", "topic": topic},
                }
                stmt = insert(Embedding).values(embedding_record)
                await session.execute(stmt)

                # Mark paper as processed
                paper = await session.get(ResearchPaper, paper_id)
                if paper:
                    paper.processed = True

                total_embedded += 1
                logger.info(f"  Embedded: {topic}")

            except Exception as e:
                error_msg = f"Error generating embedding for {topic}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

            await session.commit()

    logger.info(
        f"Manual insertion complete: "
        f"{total_inserted} inserted, "
        f"{total_embedded} embedded, "
        f"{len(errors)} errors"
    )

    return {
        "total_inserted": total_inserted,
        "total_embedded": total_embedded,
        "errors": errors,
    }


if __name__ == "__main__":
    asyncio.run(insert_manual_articles())
