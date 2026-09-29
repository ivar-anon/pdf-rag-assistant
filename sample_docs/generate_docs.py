"""Generate the sample PDF corpus.

All three documents describe a FICTIONAL company, Larkspur Home Robotics, and a
fictional customer. They exist only to exercise the pipeline: a policy
handbook, a services contract and a technical manual, i.e. the three kinds of
documents clients most often want to ask questions about.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent / "pdfs"

HANDBOOK = ("Larkspur Home Robotics — Employee Handbook (2026 edition)", [
    [("h", "1. About this handbook"),
     ("p", "This handbook applies to all full-time and part-time employees of Larkspur Home Robotics (a fictional company "
           "used for demonstration purposes). It was last revised on 12 January 2026 and replaces all earlier versions."),
     ("p", "Questions about any policy should go to the People Operations team at the internal help desk. Managers may not "
           "grant exceptions to the policies in sections 3 and 5 without written approval from People Operations."),
     ("h", "2. Working hours and remote work"),
     ("p", "Core collaboration hours are 10:00 to 15:00 in the employee's local time zone. Outside core hours, employees "
           "organise their own schedule with their manager."),
     ("p", "Employees may work remotely up to three days per week. Fully remote arrangements require a signed remote work "
           "agreement and are reviewed every six months. Employees working remotely receive a one-time home office "
           "allowance of 600 USD, paid with the first salary after the agreement is signed.")],
    [("h", "3. Paid time off"),
     ("p", "Full-time employees accrue 22 days of paid vacation per calendar year, prorated for part-time contracts. "
           "Up to five unused vacation days may be carried over to the next year; carried-over days expire on 31 March."),
     ("p", "Vacation requests of more than five consecutive working days must be submitted at least 30 days in advance. "
           "Requests of five days or fewer need 7 days' notice."),
     ("p", "In addition, employees receive 10 days of paid sick leave per year. A medical certificate is required for "
           "absences longer than three consecutive days. Sick leave does not carry over."),
     ("h", "4. Parental leave"),
     ("p", "Primary caregivers receive 16 weeks of fully paid parental leave; secondary caregivers receive 6 weeks. "
           "Parental leave can be taken within 12 months of the birth or adoption and may be split into two blocks.")],
    [("h", "5. Expenses and travel"),
     ("p", "Business expenses are reimbursed when submitted with receipts within 45 days of the expense. Expenses above "
           "250 USD require prior approval from the employee's manager."),
     ("p", "Economy class is the standard for all flights. Business class is allowed only for flights longer than "
           "8 hours and requires director approval. The daily meal allowance while travelling is 70 USD."),
     ("h", "6. Information security"),
     ("p", "All laptops must use full-disk encryption and lock automatically after 5 minutes of inactivity. Multi-factor "
           "authentication is mandatory for every company system."),
     ("p", "Suspected security incidents, including lost devices, must be reported to the security team within one hour "
           "of discovery. Customer data may never be stored on personal devices or personal cloud accounts.")],
])

CONTRACT = ("Master Services Agreement — Larkspur Home Robotics and Brightwater Retail Group", [
    [("p", "This Master Services Agreement (the \"Agreement\") is entered into on 3 February 2026 between Larkspur Home "
           "Robotics (\"Provider\") and Brightwater Retail Group (\"Customer\"). Both parties are fictional and this "
           "document is a sample for demonstration purposes only."),
     ("h", "1. Scope of services"),
     ("p", "Provider will supply, install and maintain LR-200 floor-cleaning robots in the Customer's retail stores, as "
           "described in each Statement of Work (SOW). Each SOW lists the stores, the number of units and the start date."),
     ("h", "2. Term and renewal"),
     ("p", "The initial term of this Agreement is 36 months from the effective date. After the initial term, the "
           "Agreement renews automatically for successive 12-month periods unless either party gives written notice of "
           "non-renewal at least 90 days before the end of the current term.")],
    [("h", "3. Fees and payment"),
     ("p", "The Customer pays a monthly service fee of 145 USD per robot. Invoices are issued monthly in advance and are "
           "payable within 30 days of the invoice date."),
     ("p", "Late payments accrue interest of 1.5 percent per month on the overdue amount. Provider may adjust fees once "
           "per contract year by no more than 4 percent, with 60 days' written notice."),
     ("h", "4. Service levels"),
     ("p", "Provider guarantees a monthly fleet availability of 98.5 percent. Critical faults must be responded to within "
           "4 business hours and resolved within 2 business days."),
     ("p", "If monthly availability falls below 98.5 percent, the Customer receives a service credit of 5 percent of that "
           "month's fees for each full percentage point below the target, capped at 25 percent of the monthly fees.")],
    [("h", "5. Termination"),
     ("p", "Either party may terminate this Agreement for material breach if the breach is not cured within 30 days of "
           "written notice. The Customer may terminate for convenience after the first 12 months by paying an early "
           "termination fee equal to three months of service fees."),
     ("h", "6. Limitation of liability"),
     ("p", "Except for breaches of confidentiality, each party's total liability under this Agreement is limited to the "
           "fees paid by the Customer in the 12 months before the event giving rise to the claim. Neither party is liable "
           "for indirect or consequential damages."),
     ("h", "7. Confidentiality and governing law"),
     ("p", "Confidentiality obligations survive for five years after termination. This Agreement is governed by the laws "
           "of the State of Oregon, and disputes are resolved by the state courts located in Portland, Oregon.")],
])

MANUAL = ("LR-200 Floor Robot — Technical Manual (revision C)", [
    [("h", "1. Overview"),
     ("p", "The LR-200 is a commercial floor-cleaning robot for retail spaces up to 1,200 square metres per charge. It "
           "combines a rotating brush, a 4-litre clean-water tank and a 3.5-litre recovery tank. The LR-200 is a fictional "
           "product described for demonstration purposes."),
     ("table", [["Specification", "Value"], ["Dimensions (W x D x H)", "520 x 610 x 480 mm"], ["Weight (empty tanks)", "38 kg"],
                ["Battery", "LiFePO4, 25.6 V, 40 Ah"], ["Runtime per charge", "up to 3.5 hours"],
                ["Full charge time", "2 hours 15 minutes"], ["Noise level", "58 dB(A)"], ["Max slope", "6 degrees"]]),
     ("h", "2. Charging"),
     ("p", "Return the robot to its dock when the battery indicator falls below 15 percent. The robot returns to the dock "
           "automatically at 10 percent. Do not charge the robot at ambient temperatures below 5 °C or above 40 °C.")],
    [("h", "3. Error codes"),
     ("table", [["Code", "Meaning", "Action"], ["E01", "Brush jammed", "Power off, remove debris from the brush housing."],
                ["E02", "Recovery tank full", "Empty and rinse the recovery tank."],
                ["E04", "Cliff sensor blocked", "Clean the four cliff sensors with a dry cloth."],
                ["E07", "Battery over temperature", "Move the robot to a cooler area and wait 30 minutes."],
                ["E12", "Navigation lost", "Push the robot back to a mapped area and restart the task."]]),
     ("p", "If an error code repeats more than three times in one shift, log a support ticket with the robot's serial "
           "number and the time of each error.")],
    [("h", "4. Maintenance schedule"),
     ("p", "Daily: empty and rinse the recovery tank, and wipe the squeegee blades. Weekly: clean the cliff sensors and "
           "check the brush for wrapped hair or string. Every 500 operating hours: replace the brush and the squeegee "
           "blades. Every 2,000 operating hours: have a certified technician replace the drive belts."),
     ("h", "5. Warranty"),
     ("p", "The LR-200 has a 24-month limited warranty covering manufacturing defects. The battery is covered for "
           "36 months or 1,500 charge cycles, whichever comes first. The warranty does not cover damage caused by "
           "non-approved cleaning chemicals.")],
])


def build(path: Path, title: str, pages: list[list[tuple]]) -> None:
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=8)
    h = ParagraphStyle("h", parent=styles["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=6)
    t = ParagraphStyle("t", parent=styles["Title"], fontSize=17, spaceAfter=14)
    story = [Paragraph(title, t)]
    for i, page in enumerate(pages):
        if i:
            story.append(PageBreak())
        for kind, content in page:
            if kind == "h":
                story.append(Paragraph(content, h))
            elif kind == "p":
                story.append(Paragraph(content, body))
            else:
                tbl = Table(content, hAlign="LEFT", colWidths=None)
                tbl.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                                         ("GRID", (0, 0), (-1, -1), 0.4, "#999999"), ("BACKGROUND", (0, 0), (-1, 0), "#eeeeee"),
                                         ("VALIGN", (0, 0), (-1, -1), "TOP")]))
                story += [tbl, Spacer(1, 8)]

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(20 * mm, 12 * mm, f"{title} — page {doc.page}")

    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=20 * mm,
                      bottomMargin=20 * mm, title=title, author="Sample corpus (fictional)").build(
        story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build(OUT / "employee-handbook.pdf", *HANDBOOK)
    build(OUT / "services-agreement.pdf", *CONTRACT)
    build(OUT / "lr200-technical-manual.pdf", *MANUAL)
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
