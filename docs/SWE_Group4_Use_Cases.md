# SWE Group 4 — Use Cases

---

 Use Case 1: Vehicle Search

 Description
1. The customer clicks on the search button/link.
2. The customer selects desired vehicle criteria via multiple dropdowns.
3. The customer clicks on "confirm."
4. If the vehicle exists, the system will show available vehicles.
5. Customer selects vehicle to view.

 Exception Paths
1. If the vehicle doesn't exist, or the user cancels the search, return to step 2.

 Alternate Paths
1. N/A

 Prerequisites
1. Existing user login.
2. Search for vehicle button.

 Postrequisites
1. Booking the vehicle.

---

 Use Case 2: User Registration

 Description
1. The customer clicks on the "Register" or "Sign Up" button/link.
2. The system displays the registration form.
3. The customer enters required information (name, email, password, phone number).
4. The customer clicks "Submit."
5. The system validates the input and creates a new account.
6. The system sends a confirmation email to the customer.
7. The customer is redirected to the login page.

 Exception Paths
1. If the email is already registered, the system displays an error message and returns to step 3.
2. If required fields are left blank, the system highlights the missing fields and returns to step 3.
3. If the password does not meet complexity requirements, the system displays a password policy error and returns to step 3.

 Alternate Paths
1. The customer can register using a social login (Google, Facebook) instead of filling out the form manually.

 Prerequisites
1. The customer has navigated to the application.
2. The customer does not already have an account.

 Postrequisites
1. A new user account is created in the system.
2. The customer can now log in with their credentials.

---

 Use Case 3: User Login

 Description
1. The customer clicks on the "Login" button/link.
2. The system displays the login form.
3. The customer enters their email and password.
4. The customer clicks "Sign In."
5. The system validates the credentials.
6. The system redirects the customer to the home/dashboard page.

 Exception Paths
1. If the email or password is incorrect, the system displays "Invalid credentials" and returns to step 3.
2. If the account is locked after multiple failed attempts, the system displays a lockout message and provides a "Forgot Password" link.

 Alternate Paths
1. The customer can click "Forgot Password" to initiate a password reset flow.
2. The customer can log in using a social login (Google, Facebook).

 Prerequisites
1. The customer has a registered account (Use Case 2).

 Postrequisites
1. The customer is authenticated and has access to their account features.
2. A session is created for the customer.

---

 Use Case 4: Book a Vehicle

 Description
1. The customer searches for and selects a vehicle (Use Case 1).
2. The system displays the vehicle details page (make, model, year, price per day, availability).
3. The customer selects pickup date and return date.
4. The customer selects a pickup location.
5. The customer clicks "Book Now."
6. The system displays an order summary with total cost.
7. The customer confirms the booking.
8. The system processes the reservation and displays a booking confirmation with a confirmation number.

 Exception Paths
1. If the selected dates are unavailable, the system displays an error and suggests alternate dates.
2. If the customer is not logged in, the system redirects to the login page and returns to step 5 after authentication.
3. If payment fails during confirmation, the system displays a payment error and returns to step 6 (see Use Case 5).

 Alternate Paths
1. The customer can add optional extras (insurance, GPS, child seat) before confirming in step 6.
2. The customer can change the pickup/return dates before confirming.

 Prerequisites
1. Existing user login.
2. A vehicle has been selected from search results (Use Case 1).

 Postrequisites
1. A booking record is created in the system.
2. The vehicle is marked as reserved for the selected dates.
3. A confirmation email is sent to the customer.

---

 Use Case 5: Process Payment

 Description
1. The customer proceeds to checkout from the booking summary (Use Case 4, step 6).
2. The system displays the payment form.
3. The customer selects a payment method (credit card, debit card, PayPal).
4. The customer enters payment details.
5. The customer clicks "Pay Now."
6. The system processes the payment through the payment gateway.
7. The system displays a payment success message with a receipt.

 Exception Paths
1. If the payment is declined, the system displays "Payment declined" and returns to step 4.
2. If the payment gateway times out, the system displays a timeout error and offers the customer the option to retry.
3. If the card information is invalid, the system highlights the invalid fields and returns to step 4.

 Alternate Paths
1. The customer can apply a promo code or discount before clicking "Pay Now."
2. The customer can save their payment method for future bookings.

 Prerequisites
1. Existing user login.
2. A booking has been created and is pending payment (Use Case 4).

 Postrequisites
1. The payment is recorded in the system.
2. The booking status is updated from "Pending" to "Confirmed."
3. A receipt is emailed to the customer.

---

 Use Case 6: Cancel Booking

 Description
1. The customer navigates to "My Bookings" or "Booking History."
2. The system displays a list of the customer's active and past bookings.
3. The customer selects an active booking.
4. The customer clicks "Cancel Booking."
5. The system displays the cancellation policy and any applicable fees.
6. The customer confirms the cancellation.
7. The system cancels the booking and initiates a refund if applicable.

 Exception Paths
1. If the booking is past the cancellation deadline, the system displays "Cancellation not allowed" and returns to step 3.
2. If the refund process fails, the system logs the error and notifies support to process the refund manually.

 Alternate Paths
1. The customer can modify the booking dates instead of canceling (see Use Case 7).

 Prerequisites
1. Existing user login.
2. The customer has at least one active booking.

 Postrequisites
1. The booking status is updated to "Cancelled."
2. The vehicle is released and becomes available for other customers.
3. A refund is initiated to the customer's payment method (if within policy).
4. A cancellation confirmation email is sent to the customer.

---

 Use Case 7: Modify Booking

 Description
1. The customer navigates to "My Bookings."
2. The system displays a list of the customer's active bookings.
3. The customer selects a booking and clicks "Modify."
4. The system displays the booking details with editable pickup date, return date, and pickup location fields.
5. The customer updates the desired fields.
6. The customer clicks "Save Changes."
7. The system validates availability for the new dates/location.
8. The system updates the booking and displays the updated summary with any price difference.

 Exception Paths
1. If the vehicle is not available for the new dates, the system displays an error and suggests alternate dates or vehicles.
2. If the modification results in an additional charge, the system prompts the customer to confirm the price difference before saving.

 Alternate Paths
1. The customer can cancel the modification and keep the original booking.
2. The customer can cancel the booking entirely instead of modifying (Use Case 6).

 Prerequisites
1. Existing user login.
2. The customer has an active booking that is eligible for modification (before pickup date).

 Postrequisites
1. The booking record is updated with the new details.
2. An updated confirmation email is sent to the customer.
3. Any price difference is charged or refunded accordingly.

---

 Use Case 8: Return Vehicle

 Description
1. The customer returns the vehicle to the designated pickup/return location.
2. The staff member inspects the vehicle for damage and fuel level.
3. The staff member logs into the admin system and locates the booking by confirmation number.
4. The staff member updates the vehicle status to "Returned."
5. The system calculates the final charge (including late fees, fuel charges, or damage fees if applicable).
6. The system closes the booking and updates the vehicle availability.

 Exception Paths
1. If the vehicle is returned late, the system automatically adds a late return fee to the final charge.
2. If the vehicle has damage, the staff member documents the damage and the system adds a damage fee to the customer's account.
3. If the vehicle is returned to a different location than originally booked, the system adds a one-way drop-off fee.

 Alternate Paths
1. The customer can extend the rental before the return date through "My Bookings" (Use Case 7).

 Prerequisites
1. The customer has an active booking with a vehicle checked out.
2. A staff member is available at the return location.

 Postrequisites
1. The booking status is updated to "Completed."
2. The vehicle is marked as available for future bookings.
3. A final receipt is emailed to the customer.

---

 Use Case 9: View Booking History

 Description
1. The customer clicks on "My Bookings" or "Booking History" from the navigation menu.
2. The system retrieves all bookings associated with the customer's account.
3. The system displays a list of bookings with status (Active, Completed, Cancelled), vehicle info, dates, and total cost.
4. The customer selects a booking to view full details.
5. The system displays the booking detail page with vehicle info, dates, pickup location, payment summary, and receipt.

 Exception Paths
1. If the customer has no bookings, the system displays "No bookings found" with a link to search for vehicles.

 Alternate Paths
1. The customer can filter bookings by status (Active, Completed, Cancelled).
2. The customer can sort bookings by date or price.
3. From the detail view, the customer can take action on active bookings (Modify, Cancel).

 Prerequisites
1. Existing user login.

 Postrequisites
1. The customer has reviewed their booking history.

---

 Use Case 10: Admin — Add Vehicle to Fleet

 Description
1. The admin logs into the admin dashboard.
2. The admin clicks "Add Vehicle" in the Fleet Management section.
3. The system displays a form for vehicle details.
4. The admin enters vehicle information: make, model, year, color, VIN, license plate, mileage, price per day, and vehicle category.
5. The admin uploads vehicle photos.
6. The admin clicks "Save."
7. The system validates the input and adds the vehicle to the fleet database.
8. The vehicle is now available for customer searches and bookings.

 Exception Paths
1. If required fields are missing, the system highlights the empty fields and returns to step 4.
2. If the VIN or license plate already exists in the system, the system displays a duplicate error.
3. If the uploaded image exceeds the file size limit, the system displays an error and returns to step 5.

 Alternate Paths
1. The admin can save the vehicle as "Draft" (not available for bookings) and publish it later.
2. The admin can clone an existing vehicle entry to quickly add a similar vehicle.

 Prerequisites
1. Admin user login with fleet management permissions.

 Postrequisites
1. The vehicle is added to the fleet database.
2. The vehicle appears in customer search results when matching criteria are entered.

---

 Use Case 11: Admin — Manage Fleet

 Description
1. The admin logs into the admin dashboard.
2. The admin navigates to "Fleet Management."
3. The system displays a list of all vehicles with status (Available, Reserved, Checked Out, Maintenance).
4. The admin selects a vehicle to view or edit.
5. The admin can update vehicle details, change status, set price, or mark for maintenance.
6. The admin clicks "Save Changes."
7. The system updates the vehicle record.

 Exception Paths
1. If the admin attempts to delete a vehicle that has active bookings, the system blocks the deletion and displays a warning.
2. If the admin marks a vehicle for maintenance that is currently reserved, the system notifies affected customers.

 Alternate Paths
1. The admin can filter vehicles by status, category, or location.
2. The admin can bulk-update pricing for multiple vehicles at once.
3. The admin can export the fleet list as a CSV report.

 Prerequisites
1. Admin user login with fleet management permissions.
2. At least one vehicle exists in the system.

 Postrequisites
1. The vehicle records are updated in the database.
2. Changes are reflected in customer-facing search results immediately.
