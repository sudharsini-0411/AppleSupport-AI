import pandas as pd

df = pd.read_csv("data/processed/golden_review_working.csv", dtype=str)

# Snapshot existing corrections before any changes
existing_mask = df["corrected_intent"].notna() & (df["corrected_intent"].str.strip() != "")
existing_corrections = df.loc[existing_mask, "corrected_intent"].copy()
existing_count = existing_mask.sum()

VALID_INTENTS = {
    "account_issue", "app_issue", "battery_issue", "connectivity_issue",
    "device_issue", "order_delivery", "other", "payment_billing",
    "refund", "software_update", "subscription"
}

# Manual labels for blank rows — classified by reading customer_message + conversation_context
blank_labels = {
    # tweet_id : corrected_intent
    1886740: "software_update",   # "latest phone update has completely fucked my phone"
    888069:  "software_update",   # "Ever since I updated...Voice Control constantly takes over"
    1691888: "account_issue",     # "no access to iCloud tabs" — iCloud/account sync issue
    243267:  "software_update",   # "never had problems...since I moved to iOS 11"
    2422506: "account_issue",     # phishing email targeting Apple ID / account
    1080370: "battery_issue",     # "when I update...refuses to charge" — charging/battery
    2372232: "battery_issue",     # phone shut off at 1-2% — battery behaviour
    1482949: "device_issue",      # "my phone is still doing it" — context: phone issues
    337455:  "device_issue",      # "my phone freezes like 10 times per day"
    2299160: "software_update",   # reporting a bug on iOS 11.1.1 screenshot feature
    391255:  "account_issue",     # "can't do backup" — iCloud backup issue
    258904:  "app_issue",         # App Store slow/denying purchases
    2078066: "app_issue",         # tracks out of order on purchased album — iTunes/music app
    1970610: "device_issue",      # "I" letter showing as question mark — keyboard/device bug
    1679188: "device_issue",      # "iPhone completely unresponsive to touch"
    215406:  "software_update",   # "phone almost completely unusable since upgrading to 11"
    798843:  "app_issue",         # can't attach epub in iBooks — app usage issue
    1130594: "battery_issue",     # "battery...had to plug in" — battery drain
    2614201: "device_issue",      # "fix this damn glitch" — vague but device glitch context
    644226:  "connectivity_issue",# "my phone doesn't work" — can't send messages, connectivity
    2564158: "connectivity_issue",# AirPod not connecting to iPhone X — Bluetooth connectivity
    188351:  "app_issue",         # App Store not showing update/download size — app UI issue
    804914:  "software_update",   # "11.0.3 version is not good, hang during dial" — iOS bug
    1573752: "software_update",   # "need an official update specifically to fix this"
    1533077: "device_issue",      # "I button doesn't work" — hardware button issue
    1757786: "device_issue",      # "A [?] bullshit" — keyboard/autocorrect device bug
    388931:  "software_update",   # "iOS 11 = rubbish, iPad unusable"
    2422508: "device_issue",      # "FaceTime deleted itself, can't use either camera"
    458433:  "payment_billing",   # "payment card being declined when I try to update it"
    2055778: "device_issue",      # "since I updated phone freezes, app crashes"
    299688:  "device_issue",      # "iPhone won't stop freezing" after iOS update
    2057376: "software_update",   # "been to Apple store 3 times to resolve iOS 11 issue"
    1682449: "app_issue",         # Apple Music stops on notification — app behaviour
    1712438: "device_issue",      # "I️" typing issue — keyboard/autocorrect bug
    1756286: "device_issue",      # can't type "I" — keyboard autocorrect bug
    1700858: "device_issue",      # "fix the I️ from doing that box" — keyboard bug
    2041392: "device_issue",      # word replacement not fixing "I" bug
    2145681: "device_issue",      # "I️" question mark bug
    1631248: "software_update",   # "going to send an update to fix the I️ bug"
    1860774: "battery_issue",     # "had to charge phone almost 3x day on battery save mode"
    92697:   "battery_issue",     # "drops like 8 percent" — battery drain
    1707010: "software_update",   # autocorrecting "I" to "A ?" after update
    391720:  "software_update",   # "no longer got option to turn off auto-brightness #ios11"
    1673640: "software_update",   # "NEVER HAD PROBLEMS until I updated it"
    1659652: "device_issue",      # "this bug is so annoying" — keyboard/device bug
    103200:  "device_issue",      # "phone not working still" — general device issue
    112899:  "account_issue",     # iTunes deleted purchased file from library/iCloud
    2304543: "device_issue",      # Spanish: phone stuck on startup — device issue
    738238:  "software_update",   # "#iOS11" with screenshot — iOS update issue
    2210690: "software_update",   # "apps keep crashing" after iOS update
    1681585: "device_issue",      # can't type "i" — keyboard autocorrect bug
    2741974: "device_issue",      # "phone deleted all my contacts"
    1697359: "device_issue",      # "question mark boxes" — keyboard/autocorrect bug
    2299381: "device_issue",      # "running low on storage" notification incorrectly
    1161935: "software_update",   # phone stuck "updating" for over an hour
    1581455: "device_issue",      # "iPhone 6s broke 3rd time in 10 months"
    2786405: "account_issue",     # "can't access Apple ID, lost contacts/notes"
    1977174: "device_issue",      # "mac is making this sound" — hardware/device issue
    2349340: "battery_issue",     # "battery doesn't last even 2 hours" after update
    2417443: "device_issue",      # "question mark in a box autocorrect" — keyboard bug
    284936:  "software_update",   # "iOS 11.0.2 still buggy, screen not rotating"
    581947:  "software_update",   # "High Sierra is Apple's buggiest OS"
    453887:  "app_issue",         # camera video settings reverting — app settings bug
    2833086: "device_issue",      # "MacBook having issues" — device problem
    1473339: "software_update",   # "I.T" bug — iOS 11.1 autocorrect issue
    2454614: "device_issue",      # "phone freezing all day, apps not working"
    2291691: "other",             # "Almost bought it" — phishing scam awareness, vague
    2229470: "connectivity_issue",# "MacOS bluetooth problems"
    1007125: "device_issue",      # phone freezes, speaker doesn't work — device issues
    1557811: "other",             # Arabic speaker asking for help — language barrier, vague
    2058395: "app_issue",         # "iTunes app so broken" — app issue
    451360:  "device_issue",      # "iPhone 7 glass cracked" — hardware/device issue
    303120:  "device_issue",      # "freezes, app crashes, battery drain" — multiple device issues
    851895:  "software_update",   # "no notifications for iMessage/WhatsApp after iOS11"
    1621162: "device_issue",      # "iPad display unresponsive while charging" — hardware
    506583:  "battery_issue",     # "phone won't charge"
    2812589: "connectivity_issue",# "can't make or receive calls" — connectivity/carrier
    1335188: "order_delivery",    # "bought new iPhone and shipped it to San Diego" — order
    1137981: "battery_issue",     # "battery very poor" after update
    678155:  "software_update",   # "iOS 11 fucked up all my auto fill settings"
    1412601: "device_issue",      # "keyboard question mark" issue — device/autocorrect bug
    2299234: "software_update",   # "Messages app closing, autocorrect messed up" — iOS bug
    2773230: "device_issue",      # "TouchID forgot all fingerprints" — hardware/device
    859197:  "other",             # "my own fault, socket was off" — resolved, no real issue
    2170662: "device_issue",      # "fix the I️ glitch" — keyboard/autocorrect bug
    2911733: "device_issue",      # "screen just gonna go black" — hardware issue
    1921115: "battery_issue",     # "down to 63% in 2.5 hours" — battery drain
    709073:  "account_issue",     # "purchased iCloud space but phone shows low storage"
    1680880: "device_issue",      # "What's with the characters" — keyboard/autocorrect bug
    154483:  "software_update",   # "can't update 11.1.2"
    1068927: "battery_issue",     # "latest iOS update eat up my battery"
    1744884: "software_update",   # Spanish: iOS 11.1 causing freezes/call issues
    2016009: "device_issue",      # can't type "I" — keyboard autocorrect bug
    872534:  "payment_billing",   # "activating iPhone 8 plus one time payment" — purchase/payment
    2955488: "software_update",   # "upgraded iMac to High Sierra, won't start"
    688383:  "software_update",   # "updated to IOS11, TouchID doesn't work"
    1362542: "payment_billing",   # "want to change payment method" for reservation
    505134:  "device_issue",      # "iPhone 8 Plus case slips on iPhone 7 Plus" — accessory/device
    2081382: "software_update",   # "alarm no longer makes noise" after update
    1507356: "device_issue",      # "apple headphones keep breaking" — hardware/device
    1805662: "other",             # asking about auto-capitalization feature — general question
    336585:  "device_issue",      # volume rocker not working correctly — device issue
    1123662: "device_issue",      # "loudspeaker doesn't work, apps crashing" — device issue
    1466909: "app_issue",         # "hide alerts" not working with Apple Watch — app/notification
    1773209: "device_issue",      # "I glitch" — keyboard/autocorrect bug
    2619965: "device_issue",      # "problems with my 5s" — general device issues
    1901463: "software_update",   # "iOS update is ruining my phone"
    1053624: "app_issue",         # app says "not compatible" — app issue
    2973533: "software_update",   # "updated and all my VM list is gone" — post-update issue
    2138927: "software_update",   # "updated to IOS 11.1, SO slow"
    2301167: "device_issue",      # "fix the damn glitch" — vague but device context
    1617227: "app_issue",         # camera app issue recording 4K 60 — app bug
    1641338: "device_issue",      # "fix this stupid ass I️ shit" — keyboard/autocorrect bug
    2112937: "app_issue",         # "updated to iOS 11.1, all my music is gone" — Apple Music app
    2083402: "device_issue",      # "memory is full" — storage/device issue
    669538:  "app_issue",         # "podcast app has been trash, freezes" — app issue
    409698:  "app_issue",         # "ban a specific artist from radio station" — Apple Music app
    2066127: "software_update",   # "lost all contacts when I updated to iOS 11"
    1028425: "account_issue",     # "can't sign into iTunes, can't download apps" — account
    123124:  "app_issue",         # "Twitter app unusable on iOS, music stops" — app conflict
    2105271: "software_update",   # "phone won't let me text because of this update"
    1648324: "software_update",   # "phone will probably be ruined because of this update"
    2690814: "app_issue",         # "Apple Music won't open" — app issue
    404153:  "connectivity_issue",# "keeps dropping WiFi signal" after iOS update
    2697734: "software_update",   # French: iOS 11.0.3 bugs on iPhone 6
    1997183: "device_issue",      # can't type "I️" — keyboard/autocorrect bug
    749806:  "app_issue",         # "FaceTime glitches" — app issue
    1926228: "device_issue",      # "many problems: apps, battery, freeze" — device issues
    1916663: "device_issue",      # "freezing, shutting off, not connecting" — device issues
    2966588: "device_issue",      # "I" showing as "A ?" on old posts — keyboard bug
    942238:  "app_issue",         # "if I backup on iTunes, is Apple Music backed up?" — app question
    1690522: "device_issue",      # "I️" keyboard bug — device/autocorrect
    1185059: "account_issue",     # "locked out of account/phone for 65 hours" — account lockout
    430162:  "connectivity_issue",# "left AirPod not playing audio" — Bluetooth connectivity
    1617852: "software_update",   # "updated phone, still seeing question mark boxes"
    2019509: "device_issue",      # "computer doing this when I type I️" — keyboard bug
    2035623: "connectivity_issue",# "WiFi randomly turns on" — connectivity/settings
    496876:  "app_issue",         # "music and TV apps won't open, settings freezes" — app issue
    2109873: "app_issue",         # "can't delete or move apps" — app management issue
    335750:  "device_issue",      # "MacBook black screen, requires hard reboot" — device issue
    1551686: "connectivity_issue",# "VPN keeps trying to connect but never does" — connectivity
    652727:  "other",             # vague feedback about Apple Care/software — general
    1881932: "software_update",   # "iOS update causing phone to erase text message history"
    1717431: "device_issue",      # "question mark in a box" — keyboard/autocorrect bug
    2130381: "app_issue",         # "latest update screwing up Instagram app" — app issue
    1688579: "device_issue",      # "fix this ? stuff" — keyboard/autocorrect bug
    2410334: "device_issue",      # "can't get voicemail, won't download apps" — device issues
    1153894: "app_issue",         # "Music app refuses to save playlist order" — app issue
    412348:  "device_issue",      # "volume won't change unless phone is unlocked" — device bug
    448371:  "device_issue",      # "frozen" status bar area — device/UI bug
    1703931: "account_issue",     # "can't establish Apple ID settings on new iPhone X"
    2545528: "app_issue",         # feature request for Apple Music Radio — app feedback
    2770797: "software_update",   # "LOVE if iOS 11 update would keep it from crashing"
    2068215: "app_issue",         # "#Clips app removed photos from iCloud" — app issue
    940452:  "connectivity_issue",# "fix this Bluetooth shit" — Bluetooth connectivity
    433055:  "device_issue",      # "screen replacement for Apple Watch" — device/hardware
    554722:  "software_update",   # "Apple Pay Cash not available in UK in 11.2 update"
}

# Apply labels only to blank rows
new_count = 0
for idx, row in df.iterrows():
    val = row["corrected_intent"]
    is_blank = pd.isna(val) or str(val).strip() == ""
    if is_blank:
        tweet_id = int(str(row["tweet_id"]).strip())
        label = blank_labels.get(tweet_id, "other")
        df.at[idx, "corrected_intent"] = label
        new_count += 1

# Verify existing corrections are unchanged
changed = 0
for idx in existing_corrections.index:
    if df.at[idx, "corrected_intent"] != existing_corrections[idx]:
        changed += 1

# Validation
total = len(df)
remaining_blank = df["corrected_intent"].isna().sum() + (df["corrected_intent"].str.strip() == "").sum()
invalid = df[~df["corrected_intent"].isin(VALID_INTENTS)]["corrected_intent"].tolist()
distribution = df["corrected_intent"].value_counts().to_dict()

print(f"Total rows                          : {total}")
print(f"Existing corrections before process : {existing_count}")
print(f"New corrections added               : {new_count}")
print(f"Total corrected rows                : {existing_count + new_count}")
print(f"Remaining blank labels              : {remaining_blank}")
print(f"Number of existing corrections changed: {changed}")
print(f"\nCorrected intent distribution:")
for intent, count in sorted(distribution.items()):
    print(f"  {intent}: {count}")
if invalid:
    print(f"\nINVALID LABELS FOUND: {invalid}")
else:
    print("\nNo invalid labels found.")

assert total == 200, f"Expected 200 rows, got {total}"
assert remaining_blank == 0, "Blank labels remain!"
assert changed == 0, "Existing corrections were modified!"

df.to_csv("data/processed/golden_evaluation_final.csv", index=False)
print("\nSaved: data/processed/golden_evaluation_final.csv")
